"""
Linear probe: frozen encoder, pooled features, one L2-regularized linear head over every label column,
fitted with full-batch L-BFGS.

As a script, on any HF checkpoint (a directory or a hub id):
    python probe.py runs/eurobert-210m/250101120000/mlm/best -p my-probe
As a library, train_mlm.py calls run_probe() on its best checkpoint.
"""
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from loguru import logger
from sklearn import metrics as skm
from transformers import AutoModel, AutoTokenizer


@dataclass
class ProbeConfig:
    pooling: Literal['mean', 'first'] = 'mean'  # Masked mean over tokens, or the first token.
    max_seq_length: int = 512  # Truncation length for feature extraction.
    batch_size: int = 32  # Batch size for feature extraction.
    val_fraction: float = 0.2  # Share of train rows held out to choose lambda for each label column.
    lambdas: tuple[float, ...] = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)  # L2 strengths to try.
    max_iter: int = 500  # L-BFGS iterations per fit.


@dataclass
class Task:
    column: str
    type: Literal['binary', 'multiclass', 'regression']
    num_outputs: int  # 1 for binary and regression, K for multiclass


# label columns

def find_tasks(train: pa.Table, eval_: pa.Table, text_column: str):
    """Every column except the text is a label candidate; unusable ones are skipped with a reason."""
    tasks, skipped = [], {}
    for name in train.column_names:
        if name == text_column:
            continue
        dtype = train.schema.field(name).type
        if name not in eval_.column_names:
            skipped[name] = 'missing in eval'
        elif pa.types.is_boolean(dtype):
            tasks.append(Task(name, 'binary', 1))
        elif pa.types.is_integer(dtype):
            train_values, eval_values = to_float(train[name]), to_float(eval_[name])
            seen = np.unique(train_values[~np.isnan(train_values)])
            k = len(seen)
            if k < 2 or not np.array_equal(seen, np.arange(k)) or not np.isin(eval_values[~np.isnan(eval_values)], seen).all():
                skipped[name] = f'int labels must be 0..K-1 (K >= 2), all seen in train; train has {seen[:10].astype(int).tolist()}'
            else:
                tasks.append(Task(name, 'binary' if k == 2 else 'multiclass', 1 if k == 2 else k))
        elif pa.types.is_floating(dtype):
            tasks.append(Task(name, 'regression', 1))
        else:
            skipped[name] = f'unsupported type {dtype}'
    return tasks, skipped


def to_float(column: pa.ChunkedArray) -> np.ndarray:
    return pc.cast(column, pa.float64()).to_numpy(zero_copy_only=False)  # null -> nan


# features

@torch.no_grad()
def extract(encoder, tokenizer, texts, cfg: ProbeConfig, device):
    feats = []
    for i in range(0, len(texts), cfg.batch_size):
        enc = tokenizer(
            texts[i:i + cfg.batch_size],
            padding=True,
            truncation=True,
            max_length=cfg.max_seq_length,
            return_tensors='pt',
        ).to(device)
        h = encoder(**enc).last_hidden_state
        if cfg.pooling == 'mean':
            mask = enc['attention_mask'].unsqueeze(-1).to(h.dtype)
            feats.append((h * mask).sum(1) / mask.sum(1))
        else:
            feats.append(h[:, 0])  # assumes right padding
    return torch.cat(feats).float()


# linear head

def task_losses(z, y, tasks):
    """Mean loss of each label column over its non-null rows."""
    losses, start = [], 0
    for task in tasks:
        out = z[:, start:start + task.num_outputs]
        start += task.num_outputs
        target = y[task.column]
        keep = ~target.isnan()
        if task.type == 'binary':
            losses.append(F.binary_cross_entropy_with_logits(out[keep, 0], target[keep]))
        elif task.type == 'multiclass':
            losses.append(F.cross_entropy(out[keep], target[keep].long()))
        else:
            losses.append(F.mse_loss(out[keep, 0], target[keep]))
    return losses


def fit_head(x, y, tasks, lambdas, max_iter, init=None):
    """Full-batch L-BFGS. Columns are independent, so each one can carry its own lambda in one fit."""
    head = torch.nn.Linear(x.shape[1], sum(t.num_outputs for t in tasks), device=x.device)
    if init is None:
        torch.nn.init.zeros_(head.weight)  # convex problem: zero start, no seed needed
        torch.nn.init.zeros_(head.bias)
    else:
        head.load_state_dict(init.state_dict())  # warm start from the previous lambda
    lam = torch.cat([torch.full((t.num_outputs, 1), l) for t, l in zip(tasks, lambdas)]).to(x)
    optimizer = torch.optim.LBFGS(head.parameters(), max_iter=max_iter, line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        loss = sum(task_losses(head(x), y, tasks)) + (lam * head.weight.square()).sum()  # bias not penalized
        loss.backward()
        return loss

    optimizer.step(closure)
    return head


# metrics

def column_metrics(task, out, target):
    keep = ~np.isnan(target)
    out, target = out[keep], target[keep]
    if task.type == 'binary':
        prob = 1 / (1 + np.exp(-out[:, 0]))
        two_classes = len(np.unique(target)) == 2  # AUROC/AUPRC undefined otherwise
        return {
            'auroc': skm.roc_auc_score(target, prob) if two_classes else np.nan,
            'auprc': skm.average_precision_score(target, prob) if two_classes else np.nan,
        }
    if task.type == 'multiclass':
        pred = out.argmax(1)
        return {'accuracy': skm.accuracy_score(target, pred), 'macro_f1': skm.f1_score(target, pred, average='macro')}
    pred = out[:, 0]
    return {
        'mae': skm.mean_absolute_error(target, pred),
        'rmse': skm.root_mean_squared_error(target, pred),
        'r2': skm.r2_score(target, pred),
    }


def write_summary(df: pd.DataFrame, skipped: dict, path: Path, title: str):
    lines = [f'# {title}', '']
    for task_type, group in df.groupby('type', sort=False):
        table = group.groupby('column', sort=False)[['lambda', 'n_train', 'n_eval']].first().join(
            group.pivot(index='column', columns='metric', values='value')
        )
        header = ['column', *table.columns]
        lines += [f'## {task_type}', '', '| ' + ' | '.join(header) + ' |', '|' + ' --- |' * len(header)]
        lines += ['| ' + ' | '.join([name, *(f'{v:.4g}' for v in row)]) + ' |' for name, row in table.iterrows()]
        lines.append('')
    if skipped:
        lines += ['## skipped', '', *(f'- `{name}`: {reason}' for name, reason in skipped.items()), '']
    path.write_text('\n'.join(lines), encoding='utf-8')


# entry point for train_mlm.py and the CLI

def run_probe(model, train_file, eval_file, out_dir, cfg: ProbeConfig, *, text_column='text', tokenizer=None, seed=0, device=None):
    """Probe every label column; returns the metrics table, or None if no column can be probed."""
    tables = {}
    for split, file in (('train', train_file), ('eval', eval_file)):
        table = pq.read_table(file)
        texts = table[text_column].to_pylist()
        tables[split] = table.take([i for i, t in enumerate(texts) if t and t.strip()])  # drop empty documents
    tasks, skipped = find_tasks(tables['train'], tables['eval'], text_column)

    # labels as float arrays, null -> nan; hold out part of train to choose lambda
    y = {s: {t.column: to_float(table[t.column]) for t in tasks} for s, table in tables.items()}
    perm = np.random.default_rng(seed).permutation(tables['train'].num_rows)
    n_val = int(len(perm) * cfg.val_fraction)
    val_idx, fit_idx = perm[:n_val], perm[n_val:]
    for task in list(tasks):
        col = y['train'][task.column]
        if np.isnan(col[val_idx]).all() or np.isnan(col[fit_idx]).all():
            skipped[task.column] = 'no labels in the val or fit part of train'
            tasks.remove(task)
            for cols in y.values():
                del cols[task.column]
    for name, reason in skipped.items():
        logger.warning(f'Probe skips column {name}: {reason}')
    if not tasks:
        logger.info('No label columns to probe')
        return None

    # frozen encoder, features standardized with train statistics
    device = device or ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer or model)
    tokenizer.padding_side = 'right'
    encoder = AutoModel.from_pretrained(model).to(device).eval()
    x = {s: extract(encoder, tokenizer, table[text_column].to_pylist(), cfg, device) for s, table in tables.items()}
    del encoder
    x_mean, x_std = x['train'].mean(0), x['train'].std(0) + 1e-6
    x = {s: (v - x_mean) / x_std for s, v in x.items()}

    # regression targets standardized too, so one lambda grid suits every column
    y_scale = {
        t.column: (np.nanmean(y['train'][t.column]), np.nanstd(y['train'][t.column]) + 1e-6) if t.type == 'regression' else (0.0, 1.0)
        for t in tasks
    }
    yt = {
        s: {c: torch.tensor((v - y_scale[c][0]) / y_scale[c][1], dtype=torch.float32, device=device) for c, v in cols.items()}
        for s, cols in y.items()
    }

    # one fit per lambda on the fit part, strongest first for warm starts; each column picks its own lambda
    def subset(idx):
        idx = torch.as_tensor(idx, device=device)
        return x['train'][idx], {c: v[idx] for c, v in yt['train'].items()}

    (x_fit, y_fit), (x_val, y_val) = subset(fit_idx), subset(val_idx)
    grid = sorted(cfg.lambdas, reverse=True)
    val_losses, head = [], None
    for lam in grid:
        head = fit_head(x_fit, y_fit, tasks, [lam] * len(tasks), cfg.max_iter, init=head)
        with torch.no_grad():
            val_losses.append([loss.item() for loss in task_losses(head(x_val), y_val, tasks)])
    best = [grid[i] for i in np.argmin(val_losses, axis=0)]

    # refit on all of train, then evaluate on the original target scale
    head = fit_head(x['train'], yt['train'], tasks, best, cfg.max_iter)
    with torch.no_grad():
        out = head(x['eval']).cpu().numpy()
    rows, start = [], 0
    for task, lam in zip(tasks, best):
        mean, std = y_scale[task.column]
        task_out = out[:, start:start + task.num_outputs] * std + mean
        start += task.num_outputs
        counts = {
            'lambda': lam,
            'n_train': int((~np.isnan(y['train'][task.column])).sum()),
            'n_val': int((~np.isnan(y['train'][task.column][val_idx])).sum()),
            'n_eval': int((~np.isnan(y['eval'][task.column])).sum()),
        }
        for metric, value in column_metrics(task, task_out, y['eval'][task.column]).items():
            rows.append({'column': task.column, 'type': task.type, 'metric': metric, 'value': value, **counts})
    df = pd.DataFrame(rows)

    # save metrics, summary, and the head folded back to raw pooled features and the original target scale
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / 'metrics.csv', index=False)
    write_summary(df, skipped, out_dir / 'summary.md', f'Linear probe: {model}')
    weight = (head.weight / x_std).detach()
    bias = (head.bias - weight @ x_mean).detach()
    head_rows, start = [], 0
    for task, lam in zip(tasks, best):
        mean, std = y_scale[task.column]
        for k in range(task.num_outputs):
            head_rows.append({
                'column': task.column,
                'type': task.type,
                'output': k,  # class index for multiclass; 0 otherwise
                'lambda': lam,
                'bias': bias[start + k].item() * std + mean,
                'weight': (weight[start + k] * std).cpu().tolist(),
            })
        start += task.num_outputs
    pd.DataFrame(head_rows).to_parquet(out_dir / 'head.parquet', index=False)
    logger.info(f'Probe results in {out_dir}\n{df.to_string(index=False)}')
    return df


if __name__ == '__main__':
    import tyro
    from dotenv import load_dotenv

    @dataclass
    class Args:
        model: tyro.conf.Positional[str]  # HF checkpoint directory or hub id.
        project_name: Annotated[str | None, tyro.conf.arg(aliases=['-p'])] = None  # Output folder name. Defaults to yymmddHHMMSS.
        output_root: Annotated[str, tyro.conf.arg(aliases=['-o'])] = 'runs'  # Writes to <output_root>/<project_name>.
        tokenizer: str | None = None  # Tokenizer directory or hub id, only if not stored with the model.
        seed: Annotated[int | None, tyro.conf.arg(aliases=['-s'])] = None  # Seed for the val split. Defaults to $SEED.

        # data
        train_file: str | None = None  # Parquet: text plus label columns. Defaults to $PROBE_TRAIN_FILE.
        eval_file: str | None = None  # Parquet: text plus label columns. Defaults to $PROBE_EVAL_FILE.
        text_column: str = 'text'  # Column holding the text; every other column is a label candidate.

        # probe
        probe: tyro.conf.OmitArgPrefixes[ProbeConfig] = field(default_factory=ProbeConfig)

    load_dotenv()
    args = tyro.cli(Args)
    args.train_file = args.train_file or os.environ['PROBE_TRAIN_FILE']
    args.eval_file = args.eval_file or os.environ['PROBE_EVAL_FILE']
    args.seed = args.seed if args.seed is not None else int(os.environ['SEED'])
    args.project_name = args.project_name or time.strftime('%y%m%d%H%M%S')

    out_dir = Path(args.output_root) / args.project_name
    out_dir.mkdir(parents=True)  # fails if this project already exists: never overwrite
    logger.add(out_dir / 'probe.log', level='INFO')
    (out_dir / 'args.json').write_text(json.dumps(asdict(args), indent=2), encoding='utf-8')
    run_probe(
        args.model,
        args.train_file,
        args.eval_file,
        out_dir,
        args.probe,
        text_column=args.text_column,
        tokenizer=args.tokenizer,
        seed=args.seed,
    )
