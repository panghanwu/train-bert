"""
Tokenizer-only check: how much of the ED text each model actually sees.

    python unk_check.py data/probe_eval.parquet -n 10000

UNK columns are n/a for byte-level tokenizers (no unk token: every character is encodable).
"Han as 1 token" is the share of distinct Han characters in the sample that encode to exactly one non-UNK token.
"""
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import pyarrow.parquet as pq
import tyro
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_bert


@dataclass
class Args:
    file: tyro.conf.Positional[str]  # Parquet with a text column.
    n: Annotated[int, tyro.conf.arg(aliases=['-n'])] = 10000  # Records to sample.
    bert_ids: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9)  # Entries in berts.yml.


args = tyro.cli(Args)
texts = pq.read_table(args.file, columns=['text'])['text'].to_pylist()
texts = [t for t in texts[:: max(len(texts) // args.n, 1)] if t][: args.n]  # evenly spaced sample
han_in_corpus = sorted(set(re.findall(r'[\u4e00-\u9fff]', ''.join(texts))))

print(f'{len(texts):,} records, {len(han_in_corpus):,} distinct Han chars in sample')
print(f'{"model":32s}{"UNK / tokens":>14s}{"docs with UNK":>15s}{"Han as 1 token":>16s}')
for bid in args.bert_ids:
    bert = load_bert(bid)
    tok = AutoTokenizer.from_pretrained(bert.tokenizer_source)
    unk = tok.unk_token_id
    enc = tok(texts, add_special_tokens=False)['input_ids']
    if unk is None:
        unk_rate = docs_unk = 'n/a'
    else:
        n_tok = sum(len(e) for e in enc)
        unk_rate = f'{sum(e.count(unk) for e in enc) / n_tok:.1%}'
        docs_unk = f'{sum(unk in e for e in enc) / len(enc):.1%}'
    char_ids = tok(han_in_corpus, add_special_tokens=False)['input_ids']
    one_token = sum(len(ids) == 1 and ids[0] != unk for ids in char_ids) / max(len(han_in_corpus), 1)
    print(f'{bert.name:32s}{unk_rate:>14s}{docs_unk:>15s}{one_token:>16.1%}')