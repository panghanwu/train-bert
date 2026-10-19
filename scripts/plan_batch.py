"""
Measure, for each BERT in berts.yml, the per-GPU batch that fits, the time per step, and the probe speed;
then suggest batch_size and grad_accum_steps for a target effective batch, with time estimates.

    CUDA_VISIBLE_DEVICES=1 python scripts/plan_batch.py -b 1 2 3 4 5 6 7 8 9

Runs on one GPU and extrapolates to --num-gpus, ignoring DDP communication, so times are a lower bound.
Each trial runs in a fresh process, so an out-of-memory error cannot distort the next measurement.
Training is timed on random token ids of full block length, which costs the same as real text.
"""
import math
import multiprocessing as mp
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Literal

import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
import tyro
from dotenv import load_dotenv
from loguru import logger
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))
from common import load_bert  # noqa: E402
from probe import ProbeConfig, extract  # noqa: E402

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


@dataclass
class Args:
    bert_ids: Annotated[list[int], tyro.conf.arg(aliases=['-b'])]  # Entry ids in berts.yml.
    output_root: Annotated[str, tyro.conf.arg(aliases=['-o'])] = 'runs'  # Writes to <output_root>/plan-batch-<yymmddHHMMSS>.

    # data
    mlm_train_file: str | None = None  # Defaults to $MLM_TRAIN_FILE.
    probe_train_file: str | None = None  # Defaults to $PROBE_TRAIN_FILE.
    probe_eval_file: str | None = None  # Defaults to $PROBE_EVAL_FILE.
    text_column: str = 'text'  # Column holding the text.
    sample_docs: int = 4096  # Documents sampled to measure tokens per char and probe speed.

    # training, as train_mlm.py will run it
    max_seq_length: int = 512  # Block size, capped by the tokenizer limit.
    mixed_precision: Literal['no', 'bf16'] = 'bf16'  # Same as train_mlm.py --mixed-precision.
    target_batch: int = 256  # Effective batch in blocks per optimizer step, over all GPUs.
    num_gpus: int = 8  # GPUs the real run will use.
    steps: int = 5  # Timed training steps per trial, after 2 warm-up steps.

    # probe, as probe.py will run it
    probe_batch_size: int = 256  # Same as --probe.batch-size.
    probe_max_seq_length: int = 512  # Same as --probe.max-seq-length.


def run_isolated(fn, *args):
    """Run fn(*args, queue) in a fresh process; a crash or OOM comes back as {'ok': False}."""
    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    proc = ctx.Process(target=fn, args=(*args, queue))
    proc.start()
    proc.join()
    return queue.get() if not queue.empty() else {'ok': False, 'error': f'exit code {proc.exitcode}'}


def train_trial(hf_id, block, batch_size, precision, steps, queue):
    try:
        model = AutoModelForMaskedLM.from_pretrained(hf_id).to(DEVICE).train()
        optimizer = torch.optim.AdamW(
            [
                {'params': [p for p in model.parameters() if p.ndim >= 2]},
                {'params': [p for p in model.parameters() if p.ndim < 2], 'weight_decay': 0.0},
            ],
            lr=1e-5,
        )
        ids = torch.randint(model.config.vocab_size, (batch_size, block), device=DEVICE)
        labels = ids.masked_fill(torch.rand(ids.shape, device=DEVICE) > 0.15, -100)
        if DEVICE == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        for i in range(2 + steps):  # optimizer state is allocated on the first step, so warm-up covers it
            if i == 2:
                if DEVICE == 'cuda':
                    torch.cuda.synchronize()
                start = time.perf_counter()
            with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=precision == 'bf16'):
                loss = model(input_ids=ids, labels=labels).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if DEVICE == 'cuda':
            torch.cuda.synchronize()
        queue.put({
            'ok': True,
            'sec_per_step': (time.perf_counter() - start) / steps,
            'peak_gb': torch.cuda.max_memory_allocated() / 1e9 if DEVICE == 'cuda' else float('nan'),
        })
    except torch.OutOfMemoryError:
        queue.put({'ok': False, 'error': 'out of memory'})


def probe_trial(model_id, tokenizer_id, texts, cfg, queue):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    tokenizer.padding_side = 'right'
    encoder = AutoModel.from_pretrained(model_id).to(DEVICE).eval()
    extract(encoder, tokenizer, texts[:cfg.batch_size], cfg, DEVICE)  # warm-up
    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    extract(encoder, tokenizer, texts, cfg, DEVICE)
    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    queue.put({'ok': True, 'docs_per_sec': len(texts) / (time.perf_counter() - start)})


def plan_one(bert_id, args, texts, total_chars, n_probe_docs):
    bert = load_bert(bert_id)
    tokenizer = AutoTokenizer.from_pretrained(bert.tokenizer_source)
    block = min(args.max_seq_length, tokenizer.model_max_length)
    sample_chars = sum(map(len, texts))
    sample_tokens = sum(map(len, tokenizer(texts)['input_ids']))  # special tokens included, as in train_mlm.py
    est_tokens = total_chars * sample_tokens / sample_chars
    row = {
        'bert_id': bert.id,
        'name': bert.name,
        'chars_per_token': sample_chars / sample_tokens,
        'est_tokens': est_tokens,
        'est_blocks': est_tokens / block,
    }

    # largest power-of-two batch up to the per-GPU share that trains without OOM
    per_gpu = args.target_batch // args.num_gpus
    batch_size, trial = per_gpu, {'ok': False}
    while batch_size >= 1:
        trial = run_isolated(train_trial, bert.hf_id, block, batch_size, args.mixed_precision, args.steps)
        logger.info(f'{bert.name}: batch_size={batch_size} -> {trial}')
        if trial['ok']:
            break
        batch_size //= 2
    if not trial['ok']:
        return row | {'error': trial.get('error')}

    grad_accum = per_gpu // batch_size
    steps_per_epoch = math.ceil(row['est_blocks'] / args.target_batch)
    cfg = ProbeConfig(batch_size=args.probe_batch_size, max_seq_length=args.probe_max_seq_length)
    probe = run_isolated(probe_trial, bert.hf_id, bert.tokenizer_source, texts, cfg)
    logger.info(f'{bert.name}: probe -> {probe}')
    return row | {
        'batch_size': batch_size,
        'grad_accum_steps': grad_accum,
        'peak_gb': trial['peak_gb'],
        'sec_per_step': trial['sec_per_step'],
        'steps_per_epoch': steps_per_epoch,
        'est_epoch_hours': steps_per_epoch * grad_accum * trial['sec_per_step'] / 3600,
        'probe_docs_per_sec': probe.get('docs_per_sec'),
        'est_probe_minutes': n_probe_docs / probe['docs_per_sec'] / 60 if probe['ok'] else None,
    }


if __name__ == '__main__':  # required: spawned trial processes re-import this file
    load_dotenv(REPO_DIR / '.env')
    args = tyro.cli(Args)
    args.mlm_train_file = args.mlm_train_file or os.environ['MLM_TRAIN_FILE']
    args.probe_train_file = args.probe_train_file or os.environ['PROBE_TRAIN_FILE']
    args.probe_eval_file = args.probe_eval_file or os.environ['PROBE_EVAL_FILE']
    assert args.target_batch % args.num_gpus == 0, '--target-batch must be divisible by --num-gpus'

    out_dir = Path(args.output_root) / f'plan-batch-{time.strftime("%y%m%d%H%M%S")}'
    out_dir.mkdir(parents=True)
    logger.add(out_dir / 'plan_batch.log', level='INFO')
    logger.info(f'{asdict(args)}; device={DEVICE}')

    # corpus size and a text sample, shared by every model
    column = pq.read_table(args.mlm_train_file, columns=[args.text_column])[args.text_column]
    total_chars = pc.sum(pc.utf8_length(column)).as_py()
    texts = [t for t in column.slice(0, args.sample_docs * 2).to_pylist() if t and t.strip()][:args.sample_docs]
    n_probe_docs = sum(pq.read_metadata(f).num_rows for f in (args.probe_train_file, args.probe_eval_file))
    logger.info(f'MLM train: {total_chars:,} chars; probe: {n_probe_docs:,} docs; sample: {len(texts)} docs')

    rows = []
    for bert_id in args.bert_ids:
        rows.append(plan_one(bert_id, args, texts, total_chars, n_probe_docs))
        pd.DataFrame(rows).to_csv(out_dir / 'plan.csv', index=False)  # rewritten after each model
    logger.info(f'Plan in {out_dir / "plan.csv"}\n{pd.DataFrame(rows).to_string(index=False)}')
