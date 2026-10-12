# Train BERTs

Continue MLM pretraining of the Hugging Face BERT models listed in `berts.yml` on your own text, with built-in linear probing to evaluate them.

1. **MLM continued pretraining** (`train_mlm.py`): continue masked-language-model training of an encoder listed in `berts.yml` on your corpus.
2. **Linear probing** (`probe.py`): freeze an encoder, pool its token features, and fit one L2-regularized linear layer per label column. The scores show how much of each label the representation already encodes.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in the seed and data paths
```

## Usage

Encoders are picked by `id` from `berts.yml`. Every script lists its options with `--help`.
Data paths and the seed default to `.env` (see `.env.example`).

### MLM continued pretraining

```bash
python train_mlm.py -b 7                               # single GPU
accelerate launch --multi_gpu train_mlm.py -b 7        # all GPUs, DDP
python train_mlm.py -b 7 -r first-try --probe.pooling first
```

To run on specific GPUs, list them in `CUDA_VISIBLE_DEVICES`. With several GPUs, `--num_processes` must match the count:

```bash
CUDA_VISIBLE_DEVICES=1 python train_mlm.py -b 7
CUDA_VISIBLE_DEVICES=0,2 accelerate launch --multi_gpu --num_processes 2 train_mlm.py -b 7
```

The best checkpoint (lowest eval loss) is kept; older ones are overwritten. Each run writes to a new folder and never overwrites one:

```
runs/<bert>/<run_name>/      # run_name defaults to yymmddHHMMSS
├── mlm/
│   ├── args.json
│   ├── train.log
│   ├── history.jsonl        # train and eval records
│   ├── history.png          # loss curves, redrawn at every eval
│   └── best/                # checkpoint and tokenizer; best.json records the step
└── probe/                   # only when the data has label columns
```

### Linear probing

Probe any Hugging Face checkpoint, a local directory or a hub id:

```bash
python probe.py runs/eurobert-210m/first-try/mlm/best -p eurobert-mlm
python probe.py EuroBERT/EuroBERT-210m -p eurobert-hub
python probe.py ckiplab/bert-base-chinese --tokenizer bert-base-chinese -p ckip-hub
```

To run on specific GPUs
```bash
CUDA_VISIBLE_DEVICES=1 python probe.py EuroBERT/EuroBERT-210m -p eurobert-hub
```

Output goes to `runs/<project_name>/`. Both the standalone and the automatic probe write:

| file | content |
|---|---|
| `metrics.csv` | One row per column and metric, with the chosen lambda and sample counts |
| `summary.md` | The same results as one table per task type |
| `head.parquet` | Linear head weights, applicable directly to raw pooled features |

Each label column picks its own L2 strength on a held-out part of train (`--val-fraction`), then is refit on all of train and scored on eval:

| task | metrics |
|---|---|
| binary | AUROC, AUPRC |
| multiclass | accuracy, macro-F1 |
| regression | MAE, RMSE, R² |

## Data preparation

Data is parquet, with train and eval in separate files. One row is one document.

| column | type | role |
|---|---|---|
| `text` | string | Document text. Empty and null rows are dropped. |
| any other column | see below | Label column for probing. |

**Every column other than the text is treated as a label.** Remove IDs and metadata before training. Label types:

| Arrow type | task |
|---|---|
| bool | binary |
| int, values 0..K-1 | binary (K = 2) or multiclass (K > 2); every class must appear in train |
| float | regression |
| anything else | skipped, with the reason logged and listed in `summary.md` |

A null label means "not annotated": that row is left out for that column only.

Keep integer labels as integers. pandas turns an int column with missing values into float, which would be probed as regression; use the nullable `Int64` dtype or write the file with pyarrow.

For probing, documents longer than `--probe.max-seq-length` tokens are truncated.