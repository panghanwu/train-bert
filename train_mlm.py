"""
MLM continued pretraining for one BERT in berts.yml.

Single GPU:  python train_mlm.py -b 7
Multi-GPU:   accelerate launch --multi_gpu train_mlm.py -b 7

If the data has label columns besides the text, the best checkpoint is linear-probed at the end (see probe.py).
"""
import json
import math
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from itertools import chain
from pathlib import Path
from typing import Annotated, Literal

import torch
import tyro
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list, set_seed
from datasets import load_dataset
from dotenv import load_dotenv
from loguru import logger
from matplotlib.figure import Figure
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling, get_scheduler

from common import load_bert
from probe import ProbeConfig, run_probe


@dataclass
class Args:
    bert_id: Annotated[int, tyro.conf.arg(aliases=['-b'])]  # Entry id in berts.yml.
    output_root: Annotated[str, tyro.conf.arg(aliases=['-o'])] = 'runs'  # Writes to <output_root>/<bert>/<run_name>.
    run_name: Annotated[str | None, tyro.conf.arg(aliases=['-r'])] = None  # Run folder name. Defaults to yymmddHHMMSS.
    seed: Annotated[int | None, tyro.conf.arg(aliases=['-s'])] = None  # Random seed. Defaults to $SEED.

    # data
    train_file: str | None = None  # Parquet, one document per row, optional label columns. Defaults to $MLM_TRAIN_FILE.
    eval_file: str | None = None  # Parquet, one document per row, optional label columns. Defaults to $MLM_EVAL_FILE.
    text_column: str = 'text'  # Column holding the document text.
    max_seq_length: int = 512  # Block size after concatenation, capped by the tokenizer limit.
    num_proc: int | None = None  # Processes for tokenization. None runs in the main process.
    clean_data_cache: bool = False  # After training, delete map caches; the shared parquet conversion is kept.

    # training
    mlm_probability: Annotated[float, tyro.conf.arg(aliases=['-m'])] = 0.15  # Fraction of tokens to predict.
    num_epochs: Annotated[int, tyro.conf.arg(aliases=['-e'])] = 3  # Training epochs.
    batch_size: int = 8  # Per-device batch size.
    grad_accum_steps: int = 1  # Gradient accumulation steps.
    learning_rate: float = 5e-5  # Peak learning rate.
    weight_decay: float = 0.0  # AdamW weight decay, skipped for biases and norm weights.
    lr_scheduler: Literal['linear', 'cosine', 'constant_with_warmup'] = 'linear'  # LR schedule after warmup.
    warmup_ratio: float = 0.0  # Fraction of optimizer steps used for linear warmup.
    mixed_precision: Literal['no', 'fp16', 'bf16'] = 'no'  # Use bf16 on GPUs that support it.
    max_grad_norm: float = 1.0  # Gradient clipping threshold.

    # logging and checkpoint
    logging_steps: int = 50  # Write train records to history.jsonl every N optimizer steps.
    eval_steps: int = 500  # Evaluate every N optimizer steps; keep the checkpoint only if eval loss improves.

    # probe, run on the best checkpoint when the data has label columns
    probe: ProbeConfig = field(default_factory=ProbeConfig)


load_dotenv()
args = tyro.cli(Args)
args.train_file = args.train_file or os.environ['MLM_TRAIN_FILE']
args.eval_file = args.eval_file or os.environ['MLM_EVAL_FILE']
args.seed = args.seed if args.seed is not None else int(os.environ['SEED'])
bert = load_bert(args.bert_id)

# accelerator; the main process picks the run name so every process writes to the same folder
accelerator = Accelerator(mixed_precision=args.mixed_precision, gradient_accumulation_steps=args.grad_accum_steps)
args.run_name = broadcast_object_list([args.run_name or time.strftime('%y%m%d%H%M%S')])[0]
output_dir = Path(args.output_root) / bert.name / args.run_name / 'mlm'
best_dir = output_dir / 'best'
history_path = output_dir / 'history.jsonl'

# logging
logger.remove()  # only the main process logs
if accelerator.is_main_process:
    output_dir.mkdir(parents=True)  # fails if this run already exists: never overwrite
    logger.add(sys.stderr, level='INFO')
    logger.add(output_dir / 'train.log', level='INFO')
    (output_dir / 'args.json').write_text(json.dumps(asdict(args), indent=2), encoding='utf-8')
set_seed(args.seed)
logger.info(f'BERT {bert.id}: {bert.name} ({bert.hf_id})')
logger.info(
    f'Device: {accelerator.device}, processes={accelerator.num_processes}, '
    f'distributed={accelerator.distributed_type.value}, mixed_precision={accelerator.mixed_precision}'
)

# tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(bert.tokenizer_source)
model = AutoModelForMaskedLM.from_pretrained(bert.hf_id)
block_size = min(args.max_seq_length, tokenizer.model_max_length)
logger.info(f'Special tokens: {tokenizer.special_tokens_map}')
logger.info(f'Sequence length set to {block_size} tokens')

# data: tokenize, then concatenate and cut into fixed-size blocks
raw = load_dataset('parquet', data_files={'train': args.train_file, 'eval': args.eval_file})
raw = raw.filter(lambda x: bool((x[args.text_column] or '').strip()))  # drop empty documents


def tokenize(batch):
    return tokenizer(batch[args.text_column], return_special_tokens_mask=True)


def group_texts(batch):
    concat = {k: list(chain.from_iterable(v)) for k, v in batch.items()}
    total = len(concat['input_ids']) // block_size * block_size  # drop the tail of each map batch
    return {k: [v[i:i + block_size] for i in range(0, total, block_size)] for k, v in concat.items()}


with accelerator.main_process_first():
    lm_ds = raw.map(tokenize, batched=True, num_proc=args.num_proc, remove_columns=raw['train'].column_names)
    lm_ds = lm_ds.map(group_texts, batched=True, num_proc=args.num_proc)

train_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=args.mlm_probability)
eval_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm_probability=args.mlm_probability, seed=args.seed
)  # eval collator has its own seed, so every evaluation masks the same tokens
train_dl = DataLoader(lm_ds['train'], batch_size=args.batch_size, shuffle=True, collate_fn=train_collator)
eval_dl = DataLoader(lm_ds['eval'], batch_size=args.batch_size, collate_fn=eval_collator)

# optimizer and scheduler
optimizer = torch.optim.AdamW(
    [
        # weight matrix
        {'params': [p for p in model.parameters() if p.ndim >= 2], 'weight_decay': args.weight_decay},
        # bias and norm layer
        {'params': [p for p in model.parameters() if p.ndim < 2], 'weight_decay': 0.0},
    ],
    lr=args.learning_rate,
)
model, optimizer, train_dl, eval_dl = accelerator.prepare(model, optimizer, train_dl, eval_dl)

steps_per_epoch = math.ceil(len(train_dl) / args.grad_accum_steps)
max_steps = args.num_epochs * steps_per_epoch
scheduler = get_scheduler(  # a prepared scheduler steps once per process, hence * num_processes
    args.lr_scheduler,
    optimizer,
    num_warmup_steps=int(args.warmup_ratio * max_steps) * accelerator.num_processes,
    num_training_steps=max_steps * accelerator.num_processes,
)
scheduler = accelerator.prepare(scheduler)
logger.info(f'Grouped dataset size: train={len(lm_ds["train"])}, eval={len(lm_ds["eval"])}')
logger.info(f'Optimizer steps={max_steps}')


def log_history(record):
    logger.info(record)
    if accelerator.is_main_process:
        with history_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')


def evaluate():
    model.eval()
    eval_collator.generator = None  # re-created from the seed on the next batch
    losses = []
    for batch in eval_dl:
        with torch.no_grad():
            loss = model(**batch).loss
        losses.append(accelerator.gather_for_metrics(loss.repeat(len(batch['input_ids']))))
    model.train()
    return torch.cat(losses).mean().item()


def save_best(record):
    tmp_dir = output_dir / 'best.tmp'  # write aside, then swap, so a crash never leaves a half-written best
    shutil.rmtree(tmp_dir, ignore_errors=True)
    accelerator.unwrap_model(model).save_pretrained(tmp_dir)
    tokenizer.save_pretrained(tmp_dir)
    (tmp_dir / 'best.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    shutil.rmtree(best_dir, ignore_errors=True)
    tmp_dir.rename(best_dir)


def plot_history():
    records = [json.loads(line) for line in history_path.read_text(encoding='utf-8').splitlines()]
    train = [r for r in records if 'train/loss' in r]
    evals = [r for r in records if 'eval/loss' in r]
    best = min(evals, key=lambda r: r['eval/loss'])
    fig = Figure(figsize=(10, 4), layout='constrained')  # no pyplot: no global state, nothing to close
    ax_train, ax_eval = fig.subplots(1, 2)
    ax_train.plot([r['step'] for r in train], [r['train/loss'] for r in train])
    ax_train.set(xlabel='step', ylabel='loss', title='train')
    ax_eval.plot([r['epoch'] for r in evals], [r['eval/loss'] for r in evals], marker='o')
    ax_eval.plot(best['epoch'], best['eval/loss'], marker='*', markersize=14, color='tab:red', linestyle='none')
    ax_eval.set(xlabel='epoch', ylabel='loss', title=f'eval (best at step {best["step"]})')
    fig.savefig(output_dir / 'history.png', dpi=120)


# train
model.train()
step, running, best_loss, start_time = 0, [], math.inf, time.time()
for epoch in range(args.num_epochs):
    logger.info(f'Epoch {epoch + 1}/{args.num_epochs} starts, {steps_per_epoch} optimizer steps')
    for batch in train_dl:
        with accelerator.accumulate(model):
            loss = model(**batch).loss
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        running.append(loss.detach())
        if not accelerator.sync_gradients:
            continue

        step += 1
        if step % args.logging_steps == 0:
            log_history({
                'step': step,
                'epoch': step / steps_per_epoch,
                'train/loss': torch.stack(running).mean().item(),
                'train/lr': scheduler.get_last_lr()[0],
                'train/grad_norm': grad_norm.item(),
                'time/elapsed_min': (time.time() - start_time) / 60,
                'time/eta_min': (time.time() - start_time) / step * (max_steps - step) / 60,
            })
            running.clear()

        if step % args.eval_steps == 0 or step == max_steps:
            eval_loss = evaluate()
            is_best = eval_loss < best_loss
            record = {
                'step': step,
                'epoch': step / steps_per_epoch,
                'eval/loss': eval_loss,
                'eval/perplexity': math.exp(eval_loss),
                'eval/is_best': is_best,
            }
            log_history(record)
            if is_best:
                best_loss = eval_loss
                if accelerator.is_main_process:
                    save_best(record)
            if accelerator.is_main_process:
                plot_history()

    logger.info(
        f'Epoch {epoch + 1}/{args.num_epochs} done at step {step}, '
        f'{(time.time() - start_time) / 60:.1f} min so far, best eval/loss={best_loss:.4f}'
    )

accelerator.wait_for_everyone()
logger.info(f'Best eval/loss={best_loss:.4f}, checkpoint in {best_dir}')

# clean up map caches
if args.clean_data_cache and accelerator.is_main_process:
    removed = lm_ds.cleanup_cache_files()
    logger.info(f'Removed map cache files: {removed}')

# linear probe on the best checkpoint, main process only
model = optimizer = scheduler = None  # drop our references; free_memory drops accelerate's and empties the cache
accelerator.free_memory()
if accelerator.is_main_process:
    run_probe(
        best_dir,
        args.train_file,
        args.eval_file,
        output_dir.parent / 'probe',
        args.probe,
        text_column=args.text_column,
        seed=args.seed,
        device=accelerator.device,
    )
