"""The label store: the same quote gets the same label (Critic v2, `design/critic_v2_evidence_spec.md` section 3, rule C6). No model, no network, no clock.

    store = open_store("data/critic/labels")                    # None (or null in the config) turns it off: nothing is kept, nothing is read
    got = store.get(store.claim_key(model, version, symbol, direction, quote, shown_text))        # {"label", "reason"} | None
    store.put(key, {"label": "supports", "reason": "..."})

WHY. The Critic's model labels each quote ("supports", "weak", "unrelated", "contradicts") and each fresh source ("bearish" with a quote, or "none"). Asked the same thing twice, the same model does not always answer the
same: in the four six-month tests of 2026-11-20 about a quarter of the Critic's verdicts changed between two cold runs of one request, and a recommendation's verdict turned on a label of a quote. So a label is KEPT, by
everything that makes the question what it is, and read back the next time: the model is asked about a quote only once.

THE KEY is the sha256 of: what was asked (a claim or a fresh source), the model's id, the PROMPT's version (a hash of the prompt's own text: `prompt_version`, so a changed prompt asks again), the symbol and the direction the
prompt names, the claim's quote (or the fresh source's heading), the hash of the SOURCE TEXT THE MODEL WAS SHOWN (another text asks again) and the control's result. A label is written ONLY from a reply whose control claim was read
right (the planted, unrelated notice was called unrelated): a reply that failed its control is thrown away and keeps nothing, so a failed control asks again next time. The control still runs in every call that is made; a call
with nothing left to ask (the store answered every claim and every fresh source) is not made.

THE FILES. One small JSON file per label under `folder/<first two hex>/<key>.json` (gitignored: `data/critic/` is a result of this machine's runs), written atomically (a temp file in the same folder, then a rename) so that the
Critic's records, which are checked at once, never see half a file. A file that cannot be read is a miss, never an error: the model is asked. The store never holds article text: the key is a hash, and the value is the label, the
model's short reason and, for a bearish fresh source, the one sentence the model copied from it.
"""
import hashlib
import json
import os
import tempfile
from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path

from prometheus.rag.config import REPO_ROOT

CONTROL_PASSED = "control_passed"      # the only control result a label is written with (a failed control keeps nothing)
CLAIM, FRESH = "claim", "fresh"


@lru_cache(maxsize=8)
def prompt_version(max_words: int) -> str:
    """A short hash of the Critic's prompt as it reads for `max_words`: the system message, the retry message and the shape of the user message. Change a word of the prompt and the version changes, so no label made with the old
    one is read back."""
    from prometheus.prompts import critic as prompts
    text = "\n".join([prompts.review_system(date(2000, 1, 1), "SYM", "Company", "bullish", max_words), prompts.RETRY_REVIEW,
                      prompts.review_message("SYM", "bullish", [("[1]", "quote")], [("[1]", "heading", "text")], [("F1", "heading", "text")])])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _sha(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


class LabelStore:
    def __init__(self, folder: str | Path):
        """`folder`: where the labels are kept (made on the first write). A relative path is from the repository."""
        path = Path(folder)
        self.folder = path if path.is_absolute() else REPO_ROOT / path

    # ---------------------------------------------------------------- the keys

    @staticmethod
    def claim_key(model: str | None, version: str, symbol: str, direction: str, quote: str, shown_text: str) -> str:
        """The key of one claim's label: the model, the prompt's version, the symbol, the direction, the quote and the hash of the text the model was shown."""
        return _sha(CLAIM, str(model), version, symbol, direction, quote, _sha(shown_text), CONTROL_PASSED)

    @staticmethod
    def fresh_key(model: str | None, version: str, symbol: str, direction: str, heading: str, shown_text: str) -> str:
        """The key of one fresh source's label: the model, the prompt's version, the symbol, the direction, the heading the model read and the hash of the text it was shown."""
        return _sha(FRESH, str(model), version, symbol, direction, heading, _sha(shown_text), CONTROL_PASSED)

    # ---------------------------------------------------------------- the files

    def _path(self, key: str) -> Path:
        return self.folder / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict | None:
        """What `put` kept under `key`, or None: there is none, or the file cannot be read."""
        try:
            kept = json.loads(self._path(key).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return kept if isinstance(kept, dict) and isinstance(kept.get("label"), str) else None

    def put(self, key: str, value: dict) -> bool:
        """Keep a label (`{"label", "reason"}` for a claim, `{"label", "quote"}` for a fresh source). Atomic. A store that cannot be written (a read-only disk) is not an error: False, and the next review asks the model again."""
        path = self._path(key)
        body = {**value, "stored_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle, temp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
            try:
                with os.fdopen(handle, "w", encoding="utf-8") as out:
                    json.dump(body, out, ensure_ascii=False)
                    out.flush()
                os.replace(temp, path)
            except BaseException:
                try:
                    os.unlink(temp)
                except OSError:
                    pass
                raise
        except OSError:
            return False
        return True

    def count(self) -> int:
        """How many labels are kept (for a report)."""
        return sum(1 for _ in self.folder.glob("*/*.json")) if self.folder.is_dir() else 0


def open_store(folder: str | Path | None) -> LabelStore | None:
    """The store the config names (`evidence.label_store`), or None when it is null: no label is kept or read."""
    return LabelStore(folder) if folder else None
