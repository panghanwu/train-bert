"""Look up a BERT entry in berts.yml by id."""
from dataclasses import dataclass
from pathlib import Path

import yaml

BERTS_YML = Path(__file__).with_name('berts.yml')


@dataclass(frozen=True)
class Bert:
    id: int
    name: str
    hf_id: str
    tokenizer_id: str | None = None  # Only when the tokenizer lives in another repo (e.g. CKIP).
    description: str = ''

    @property
    def tokenizer_source(self) -> str:
        return self.tokenizer_id or self.hf_id


def load_bert(bert_id: int) -> Bert:
    berts = [Bert(**entry) for entry in yaml.safe_load(BERTS_YML.read_text(encoding='utf-8'))]
    for bert in berts:
        if bert.id == bert_id:
            return bert
    raise KeyError(f'bert_id {bert_id} not in {BERTS_YML.name}; available ids: {[b.id for b in berts]}')
