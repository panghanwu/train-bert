"""
Tiny synthetic parquet for smoke-testing train_mlm.py and probe.py, plus a matching .env.smoke.

    python scripts/make_smoke_test.py
    python scripts/make_smoke_test.py -o /tmp/smoke --env-file /tmp/.env.smoke

Label columns cover every probe path:
    is_pos  bool          -> binary
    grade   int64 0..2    -> multiclass
    score   float64       -> regression
    flag    int64 + nulls -> binary, null rows left out for this column only
    note    string        -> skipped (unsupported type)
A few empty and null texts check that empty documents are dropped.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tyro

REPO_DIR = Path(__file__).resolve().parents[1]


@dataclass
class Args:
    out_dir: Annotated[Path, tyro.conf.arg(aliases=['-o'])] = REPO_DIR / 'data' / 'smoke-test'  # Parquet output folder.
    env_file: Path = REPO_DIR / '.env.smoke'  # Env file pointing MLM_* and PROBE_* at the generated parquet.
    seed: Annotated[int, tyro.conf.arg(aliases=['-s'])] = 3345678  # Data seed, also written as SEED.
    n_train: int = 200  # Non-empty train documents.
    n_eval: int = 60  # Non-empty eval documents.


SUBJECTS = ['the patient', 'my neighbor', 'the new model', 'our team', 'the old printer', 'this recipe', 'the city bus']
POSITIVE = ['works really good', 'feels good today', 'gave a good result', 'did a good job', 'looks good overall']
NEGATIVE = ['broke down again', 'feels bad today', 'gave a poor result', 'did a terrible job', 'looks bad overall']
TOPICS = [  # grade 0..2 is the topic of the trailing clause
    ['during the morning walk', 'after a long weekend', 'before the holiday trip'],
    ['in the hospital ward', 'at the clinic visit', 'after the blood test'],
    ['while running the code', 'during the model training', 'after the software update'],
]
FILLERS = ['and nobody was surprised', 'as everyone expected', 'which was worth noting', 'so we wrote it down', 'and then we moved on']


def make_split(n: int, rng: np.random.Generator) -> pa.Table:
    texts, is_pos, grade, score, flag, note = [], [], [], [], [], []
    for i in range(n):
        pos = bool(rng.random() < 0.5)
        g = int(rng.integers(3))
        n_fill = int(rng.integers(1, 5))
        words = [
            rng.choice(SUBJECTS),
            rng.choice(POSITIVE if pos else NEGATIVE),
            rng.choice(TOPICS[g]),
            *rng.choice(FILLERS, n_fill),
        ]
        texts.append(' '.join(words) + '.')
        is_pos.append(pos)
        grade.append(g)
        score.append(float(n_fill + 0.5 * pos + rng.normal(0, 0.3)))  # tied to text length and sentiment
        flag.append(None if rng.random() < 0.3 else int(pos))  # ~30% unannotated
        note.append(f'row-{i}')

    # empty and null documents, expected to be dropped
    for t in ['', '   ', None]:
        texts.append(t)
        is_pos.append(False)
        grade.append(0)
        score.append(0.0)
        flag.append(0)
        note.append('empty')

    return pa.table({
        'text': pa.array(texts, pa.string()),
        'is_pos': pa.array(is_pos, pa.bool_()),
        'grade': pa.array(grade, pa.int64()),
        'score': pa.array(score, pa.float64()),
        'flag': pa.array(flag, pa.int64()),  # pyarrow keeps nulls in an int column
        'note': pa.array(note, pa.string()),
    })


args = tyro.cli(Args)
out_dir = args.out_dir.resolve()
out_dir.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(args.seed)
paths = {}
for split, n in (('train', args.n_train), ('eval', args.n_eval)):
    table = make_split(n, rng)
    paths[split] = out_dir / f'{split}.parquet'
    pq.write_table(table, paths[split])
    print(f'{split}: {table.num_rows} rows -> {paths[split]}')

args.env_file.write_text(
    f"""# Smoke test data from scripts/make_smoke_test.py
SEED={args.seed}

MLM_TRAIN_FILE={paths['train']}
MLM_EVAL_FILE={paths['eval']}

PROBE_TRAIN_FILE={paths['train']}
PROBE_EVAL_FILE={paths['eval']}
""",
    encoding='utf-8',
)
print(f'env -> {args.env_file}')
