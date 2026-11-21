"""The Proving Ground's results as FILES (design/proving_ground_spec.md section 5, P6; design choices entry 126).

    store = ProvingStore()                                   # the root of `configs/proving_v1.json` (data/proving); `ProvingStore(tmp_path)` in a test
    store.save_run(run)                                      # runs/<run_id>.json (one `ProvingRun`, written through the one JSON encoder) and the index; atomic
    store.load_run("tm-20261120-0930")                       # a `ProvingRun`
    store.runs()                                             # the index, newest first: id, kind, status, created, prompt, range, headline, return, excess
    store.cache_put(key, work_order_dict); store.cache_get(key)       # cache/<key>.json: the research of a decision day (a later step defines the key)
    store.save_session(request_id, payload); store.load_session(request_id)       # sessions/<request_id>.json: a finished desk run for the Track Record (a later step)

    data/proving/
        runs/<run_id>.json        runs/index.json        cache/<key>.json        sessions/<request_id>.json

THE STORE WRITES ONLY HERE. The root is created on the first write (`data/proving/` is in `.gitignore`), and a root that is, or is inside, a folder the desk keeps its data in (`data/historical`, `data/processed`, `data/ledger`, `data/news`, `data/filings`,
`data/press_releases`, `data/raw`, `news`) is refused (ValueError): a Proving Ground result never lands among the stored market data, the vector store or the real ledgers (P6). A name (a run id, a cache key, a request id) is checked
(`SAFE_NAME`: letters, digits, `_`, `.`, `-`, starting with a letter or a digit) so that it cannot leave its folder. Every write is ATOMIC: a temp file in the same folder, then a rename, so a reader never sees half a file and a crash leaves the
old one. Money is written by the one encoder (`money.dumps`: a Decimal is a JSON number, never a string); a file is read back with `json.loads` as it always was (floats) and the typed models turn the numbers into Decimals at their doors
(a work order in the cache is NEVER read with `parse_float=Decimal`: the Advisor's and the Critic's numbers in it are measurements). No model, no network.
"""
import json
import os
import re
import tempfile
from pathlib import Path

from prometheus.common.money import dumps
from prometheus.proving.config import load_config
from prometheus.proving.record import ProvingRun
from prometheus.rag.config import REPO_ROOT

SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-]{0,119}")
NEVER_UNDER = tuple(REPO_ROOT / p for p in ("data/historical", "data/processed", "data/ledger", "data/news", "data/filings", "data/press_releases", "data/raw", "news"))
INDEX = "index"


class ProvingStore:
    def __init__(self, root: str | Path | None = None):
        """`root`: where to write (default: `store.root` of the config). Nothing is created until the first write. Raises ValueError for a root inside a folder of the desk's own data."""
        self.root = (Path(root) if root else load_config().store_root).resolve()
        for forbidden in NEVER_UNDER:
            if self.root == forbidden.resolve() or forbidden.resolve() in self.root.parents:
                raise ValueError(f"the Proving Ground never writes under {forbidden.relative_to(REPO_ROOT)}: {self.root}")

    # ---------------------------------------------------------------- files

    def _path(self, folder: str, name: str) -> Path:
        if not isinstance(name, str) or SAFE_NAME.fullmatch(name) is None:
            raise ValueError(f"{str(name)[:40]!r} is not a name this store keeps: letters, digits, '_', '.' and '-', starting with a letter or a digit, at most 120")
        return self.root / folder / f"{name}.json"

    def _run_path(self, run_id: str) -> Path:
        if run_id == INDEX:
            raise ValueError(f"{INDEX!r} is the name of the index of the runs, not of a run")
        return self._path("runs", run_id)

    @staticmethod
    def _write(path: Path, text: str) -> None:
        """Atomic: a temp file in the same folder, written and synced, then renamed over the file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as out:
                out.write(text)
                out.flush()
                os.fsync(out.fileno())
            os.replace(temp, path)
        except BaseException:
            try:
                os.unlink(temp)
            except OSError:
                pass
            raise

    @staticmethod
    def _read(path: Path):
        return json.loads(path.read_text(encoding="utf-8"))

    # ---------------------------------------------------------------- runs

    def save_run(self, run: ProvingRun) -> Path:
        """Write the run (`runs/<run_id>.json`) and its row of the index. Returns the file."""
        path = self._run_path(run.run_id)
        self._write(path, dumps(run.model_dump(mode="json"), indent=1))
        rows = [r for r in self.runs() if r.get("run_id") != run.run_id]
        self._write(self._path("runs", INDEX), dumps([self._row(run), *rows], indent=1))
        return path

    def load_run(self, run_id: str) -> ProvingRun:
        """One saved run. Raises FileNotFoundError when there is none, ValueError when the file is not a run."""
        path = self._run_path(run_id)
        if not path.is_file():
            raise FileNotFoundError(f"there is no saved run {run_id!r}")
        return ProvingRun.model_validate(self._read(path))

    def delete_run(self, run_id: str) -> bool:
        """Remove a saved run (the user's own result file; never a cache entry or a session) and its row of the index. False when there was none."""
        path = self._run_path(run_id)
        if not path.is_file():
            return False
        path.unlink()
        self._write(self._path("runs", INDEX), dumps([r for r in self.runs() if r.get("run_id") != run_id], indent=1))
        return True

    def runs(self) -> list[dict]:
        """The index of the saved runs, newest first (by `created_utc`): {run_id, kind, status, created_utc, prompt, range, headline, return_pct, excess_return_pct}. An index that is missing or cannot be read is rebuilt from the run files."""
        path = self._path("runs", INDEX)
        try:
            rows = self._read(path)
            if isinstance(rows, list) and all(isinstance(r, dict) and isinstance(r.get("run_id"), str) for r in rows):
                return sorted(rows, key=lambda r: str(r.get("created_utc")), reverse=True)
        except (OSError, ValueError):
            pass
        return self.rebuild_index()

    def rebuild_index(self) -> list[dict]:
        """The index made again from the run files (a file that is not a run is left out). Not written until a run is saved or deleted."""
        rows = []
        for path in sorted((self.root / "runs").glob("*.json")) if (self.root / "runs").is_dir() else []:
            if path.stem == INDEX:
                continue
            try:
                rows.append(self._row(ProvingRun.model_validate(self._read(path))))
            except (OSError, ValueError):
                continue
        return sorted(rows, key=lambda r: str(r.get("created_utc")), reverse=True)

    @staticmethod
    def _row(run: ProvingRun) -> dict:
        portfolio = run.stats.portfolio
        return {"run_id": run.run_id, "kind": run.kind, "status": run.status, "created_utc": run.created_utc, "prompt": (run.request or {}).get("prompt"),
                "range": None if run.range is None else run.range.model_dump(mode="json"), "headline": run.stats.headline.sentence,
                "return_pct": None if portfolio is None else portfolio.return_pct, "excess_return_pct": None if portfolio is None else portfolio.excess_return_pct, "speed": run.speed}

    # ---------------------------------------------------------------- the research cache and the sessions

    def cache_put(self, key: str, value: dict) -> Path:
        """Keep the research of a decision day: `value` is a plain dict (the work order as the Critic left it), written by the one encoder."""
        path = self._path("cache", key)
        self._write(path, dumps(value))
        return path

    def cache_get(self, key: str) -> dict | None:
        """What `cache_put` kept, as `json` reads it (floats), or None when there is none or the file cannot be read."""
        try:
            value = self._read(self._path("cache", key))
        except (OSError, ValueError):
            return None
        return value if isinstance(value, dict) else None

    def has_cache(self, key: str) -> bool:
        return self._path("cache", key).is_file()

    def save_session(self, request_id: str, payload: dict) -> Path:
        """Keep a finished desk run (the `--save` capture format) for the Track Record."""
        path = self._path("sessions", request_id)
        self._write(path, dumps(payload))
        return path

    def load_session(self, request_id: str) -> dict | None:
        try:
            value = self._read(self._path("sessions", request_id))
        except (OSError, ValueError):
            return None
        return value if isinstance(value, dict) else None

    def sessions(self) -> list[str]:
        """The request ids of the saved sessions, oldest file first."""
        folder = self.root / "sessions"
        return [p.stem for p in sorted(folder.glob("*.json"), key=lambda p: p.stat().st_mtime)] if folder.is_dir() else []
