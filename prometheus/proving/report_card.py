"""The report card of the Critic's rules: every finding code scored against what the recommendations did (Critic v2, `design/critic_v2_evidence_spec.md` rule C1 and section 6). PURE: no model, no network, no clock.

    card = report_card.build(runs=None)         # every saved Time Machine run, the recorded ones and the Track Record; or the `ProvingRun`s given
    print(report_card.report(card))              # the card as text
    ./test_proving.sh critic-report [--run RUN_ID ...] [--save FILE]          free

WHY. A reason the Critic denies a recommendation for must EARN its place: the denials should be recommendations that did worse than the ones it approved. In the four six-month tests of 2026-11-20 three reasons made most of the
denials and denied winners in every theme, and nobody could see it until the outcomes were laid beside the findings by hand. This is that routine: a program, run again whenever there are new outcomes, so that rule C1 ("a blocker that
denies winners is a defect to fix or demote; a new blocker needs a reason in words AND outcomes on data it was not found on") is a routine and not a memory.

WHAT IT COUNTS. One entry for each RECOMMENDATION THE CRITIC REVIEWED that has an outcome (an approved or a denied one, traded or hypothetical: the outcome's own `verdict`), once, whatever the number of runs it is in (the same `record_id`
counts once: the oldest run keeps it). Its findings are the codes of its review: the BLOCKERS, the codes of the findings that asked for a person (`human`: a denial since entry 103) and the WARNINGS. They are on the outcome's `critic`
cell for a run saved since the codes were kept there; for an older one they are read from the review in the research cache (`ReviewIndex`: the store's cache, the runs' own desks and the recorded research of
`tests/fixtures/proving_real/*_cache`), which holds every review of the days the run researched. An outcome whose review cannot be found is counted in `unmatched` and left out, never guessed.
A denial for the user's own RISK PROFILE (`SUITABILITY`: `risk_above_profile` and its three kin) is suitability, not a forecast: its rows are listed with their counts and are NEVER judged by returns, and a recommendation that carries one is
left out of every comparison (the verdict's and every other rule's `rest`).

THE NUMBERS. For each CONFIG VERSION of the Critic (the review's `audit.config_version`: version 1 and version 2 reviews are never mixed in one row) the card holds `verdict` (the denied against the approved, without the suitability
denials), `rows` (one for each finding code: `fired` recommendations it fired on, of those `closed` and `won`, and the mean excess return over SPY of the recommendations it fired on, `hit`, against all the others of that version,
`rest`; the `gap` is hit minus rest, a negative gap means the rule picks losers, which is what a denying rule is for) and `by_family` (the Critic's verdict inside each of the Advisor's strategy families). Every row also says the same per
THEME (the run's own request: the report card never guesses a sector). `gap`, its standard error `se` and `inside_chance` come from `stats.gap_of`, the one place a gap of two groups of excess returns is judged (the Critic's own sentence
of a run uses it too): the gap is inside chance when it is within `stats.CHANCE_BAND` standard errors of zero. A comparison with a group under `min_sample` (20) usable outcomes is `too_few`. The sentence of a row is one of a CLOSED set of templates
(`SENTENCES`): the numbers are the only thing put in.

HONEST LIMITS, said in `limits`: the outcomes of overlapping weeks are not independent, so the standard errors flatter a result; a rule that was chosen on a window can only be confirmed on later outcomes; and the stocks are those of a few themes.
"""
import json
from dataclasses import dataclass
from pathlib import Path

from prometheus.proving import stats as stats_mod
from prometheus.proving.config import ProvingConfig, load_config
from prometheus.proving.record import ProvingRun, RecOutcome
from prometheus.proving.store import ProvingStore
from prometheus.rag.config import REPO_ROOT

# The findings that say "this is not for the user's risk profile" (the Critic's risk-fit blockers): suitability, not a forecast, so never judged by returns and never part of a comparison.
SUITABILITY = ("risk_above_profile", "symbol_excluded", "sector_excluded", "price_below_profile_min")
KIND_ORDER = ("blocker", "human", "warning")
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "proving_real"
LIMITS = ("The outcomes of overlapping weeks are not independent of one another, so the standard errors flatter a result.",
          "A rule that was chosen on one window can only be confirmed by later outcomes: judge each rule on the weeks that came after it was set.",
          "The stocks are those of a few themes, in one stretch of the market.")

# The closed set of sentences (the numbers are the only thing put into them). `{hit}` is what the rule fired on (for the verdict: the denied), `{rest}` the others (for the verdict: the approved).
SENTENCES = {
    "denies_winners": "Denies winners: {hit} did better than {rest} by {points} points, more than chance gives with this many.",
    "picks_losers": "Picks losers: {hit} did worse than {rest} by {points} points, more than chance gives with this many.",
    "no_clear_difference": "No clear difference: a gap of {points} points is what chance alone gives with this many.",
    "too_few": "Too few to say: the smaller group has {n} with an outcome, and {min_sample} are needed.",
    "suitability": "A suitability rule (the user's own risk profile): it is listed and never judged by returns.",
}


@dataclass
class Entry:
    """One reviewed recommendation with an outcome."""
    record_id: str
    run_id: str
    theme: str
    family: str | None
    verdict: str
    excess: float | None            # None: no excess return over the benchmark (not usable in a mean)
    result: str
    version: str
    blockers: list[str]
    warnings: list[str]
    human: list[str]

    @property
    def suitability(self) -> bool:
        return any(code in SUITABILITY for code in self.blockers)

    def codes(self) -> dict[str, str]:
        """{code: kind} of what the review found: a blocker, a finding that asked for a person (a code that is also a warning is the person-asking one: it denied), or a warning."""
        found = {code: "warning" for code in self.warnings}
        found.update({code: "human" for code in self.human})
        found.update({code: "blocker" for code in self.blockers})
        return found


# ---------------------------------------------------------------- the reviews of older runs

_FILES: dict = {}      # (path, mtime, size) -> {record_id: review digest}: a research file is read once while it does not change


def _digest(review: dict) -> dict | None:
    if not isinstance(review, dict) or not isinstance(review.get("record_id"), str) or not isinstance(review.get("approved"), bool):
        return None
    codes = lambda key: [f["code"] for f in review.get(key) or [] if isinstance(f, dict) and isinstance(f.get("code"), str)]
    human = review.get("requires_human_reasons")
    return {"version": ((review.get("audit") or {}).get("config_version")) if isinstance(review.get("audit"), dict) else None, "blockers": codes("blockers"), "warnings": codes("warnings"),
            "human": [r for r in human if isinstance(r, str)] if isinstance(human, list) else []}


def _file_digests(path: Path) -> dict[str, dict]:
    try:
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size)
    except OSError:
        return {}
    if key not in _FILES:
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            body = None
        order = body.get("order") if isinstance(body, dict) else None
        reviews = order.get("reviews") if isinstance(order, dict) else None
        _FILES[key] = {r["record_id"]: d for r in (reviews if isinstance(reviews, list) else []) for d in [_digest(r)] if d is not None}
    return _FILES[key]


class ReviewIndex:
    """{record_id: the findings of its review} over the places a review is kept: the runs' own desks, the store's research cache and the recorded research of the tests. Read only when a record is asked for (one that was saved with
    its codes needs none of this)."""

    def __init__(self, runs: list[ProvingRun], folders: list[Path], desks: bool = True):
        """`desks` False reads the research folders only (the pool the spec's first table was made from: a run whose research is no longer in the cache is then left out of the older rows)."""
        self.runs, self.folders, self.desks, self._found = runs, folders, desks, None

    def _build(self) -> dict[str, dict]:
        found: dict[str, dict] = {}
        for folder in self.folders:
            for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
                for record_id, digest in _file_digests(path).items():
                    found.setdefault(record_id, digest)
        for run in self.runs if self.desks else []:
            for desk in ((run.coverage or {}).get("desks") or {}).values():
                for review in (desk or {}).get("reviews") or []:
                    digest = _digest(review)
                    if digest is not None:
                        found.setdefault(review["record_id"], digest)
        return found

    def get(self, record_id: str) -> dict | None:
        if self._found is None:
            self._found = self._build()
        return self._found.get(record_id)


def default_folders(store: ProvingStore | None) -> list[Path]:
    """The research folders: the store's cache, then the recorded research beside the real runs of the tests."""
    return [*([store.root / "cache"] if store is not None else []), *sorted(FIXTURES.glob("*_cache"))]


# ---------------------------------------------------------------- the entries

def theme_of(run: ProvingRun) -> str:
    """The run's own request, in a word: the topic the Account Manager read ("oil", "AI stocks"), else the symbols or the prompt; the Track Record's recommendations are the live sessions'."""
    if run.kind == "track":
        return "live sessions"
    request = run.request or {}
    intent = request.get("intent") if isinstance(request.get("intent"), dict) else {}
    return str(intent.get("topic") or ", ".join(intent.get("symbols") or []) or request.get("prompt") or run.run_id)


def _usable(outcome: RecOutcome) -> bool:
    return outcome.result in ("won", "lost", "open") and outcome.return_pct is not None and outcome.excess_return_pct is not None


def entries_of(runs: list[ProvingRun], index: ReviewIndex) -> tuple[list[Entry], int]:
    """(the entries, how many reviewed outcomes had no findings to be found): one entry for each recommendation the Critic reviewed that has an outcome, once whatever the number of runs it is in (the first run given keeps it)."""
    found: list[Entry] = []
    seen: set[str] = set()
    unmatched = 0
    for run in runs:
        theme = theme_of(run)
        for o in run.outcomes:
            if o.verdict not in ("approved", "denied") or not o.record_id or o.record_id in seen:
                continue
            seen.add(o.record_id)
            cell = o.critic
            if cell is not None and cell.blockers is not None and cell.warnings is not None and cell.human is not None and cell.config_version:
                version, blockers, warnings, human = cell.config_version, cell.blockers, cell.warnings, cell.human
            else:
                digest = index.get(o.record_id)
                if digest is None or not digest["version"]:
                    unmatched += 1
                    continue
                version, blockers, warnings, human = digest["version"], digest["blockers"], digest["warnings"], digest["human"]
            found.append(Entry(o.record_id, run.run_id, theme, o.advisor.family if o.advisor else None, o.verdict, o.excess_return_pct if _usable(o) else None, o.result, version, list(blockers), list(warnings), list(human)))
    return found, unmatched


# ---------------------------------------------------------------- the numbers

def group_of(entries: list[Entry]) -> dict:
    """{n (with an excess return), closed, won, mean_excess}: the numbers of one group of entries."""
    usable = [e for e in entries if e.excess is not None]
    return {"n": len(usable), "closed": sum(1 for e in entries if e.result in ("won", "lost")), "won": sum(1 for e in entries if e.result == "won"), "mean_excess": stats_mod._mean(e.excess for e in usable)}


def _points(value: float) -> str:
    return stats_mod._points(abs(value))


def sentence_of(code: str, row: dict, subject: tuple[str, str], min_sample: int) -> str:
    """The row's sentence: one template of `SENTENCES`, chosen from the numbers alone."""
    return SENTENCES[code].format(hit=subject[0], rest=subject[1], n=min(row["hit"]["n"], row["rest"]["n"]), min_sample=min_sample, points=_points(row["gap"]) if row["gap"] is not None else "")


def compare(hit: list[Entry], rest: list[Entry], min_sample: int, suitability: bool = False, subject: tuple[str, str] = ("what it fired on", "the others")) -> dict:
    """The numbers of one rule (or of the verdict) against the rest: `hit` and `rest` (each `group_of`), `gap`, `se`, `inside_chance`, `sentence_code` and `sentence`. A group under `min_sample` is `too_few` (its numbers are still shown);
    a `suitability` row is listed with its counts and never judged: no gap, no standard error."""
    gap = stats_mod.gap_of([e.excess for e in hit if e.excess is not None], [e.excess for e in rest if e.excess is not None])
    row = {"hit": group_of(hit), "rest": group_of(rest), "gap": None, "se": None, "inside_chance": None}
    if gap is not None and not suitability:
        row.update(gap=gap["gap"], se=gap["se"], inside_chance=gap["inside_chance"])
    code = ("suitability" if suitability else "too_few" if gap is None or min(row["hit"]["n"], row["rest"]["n"]) < min_sample else "no_clear_difference" if gap["inside_chance"]
            else "denies_winners" if gap["gap"] > 0 else "picks_losers")
    row["sentence_code"] = code
    row["sentence"] = sentence_of(code, row, subject, min_sample)
    return row


def _by_theme(hit: list[Entry], rest: list[Entry], themes: list[str], min_sample: int, judged: bool) -> dict:
    out = {}
    for theme in themes:
        h, r = [e for e in hit if e.theme == theme], [e for e in rest if e.theme == theme]
        if not h:
            continue
        cell = compare(h, r, min_sample, suitability=not judged)
        out[theme] = {"n": cell["hit"]["n"], "closed": cell["hit"]["closed"], "won": cell["hit"]["won"], "mean_excess": cell["hit"]["mean_excess"], "rest_n": cell["rest"]["n"], "rest_closed": cell["rest"]["closed"], "rest_won": cell["rest"]["won"],
                      "rest_mean_excess": cell["rest"]["mean_excess"],
                      "gap": cell["gap"], "se": cell["se"], "inside_chance": cell["inside_chance"], "sentence_code": cell["sentence_code"]}
    return out


def _version_card(version: str, entries: list[Entry], min_sample: int, themes: list[str]) -> dict:
    population = [e for e in entries if not e.suitability]      # a suitability denial is not a forecast: out of every comparison
    approved, denied = [e for e in population if e.verdict == "approved"], [e for e in population if e.verdict == "denied"]
    verdict = {"code": "denied", "kind": "verdict", "suitability": False, "fired": len(denied), **compare(denied, approved, min_sample, subject=("the denied", "the approved"))}
    verdict["by_theme"] = _by_theme(denied, approved, themes, min_sample, True)
    kinds: dict[str, set[str]] = {}
    for e in entries:
        for code, kind in e.codes().items():
            kinds.setdefault(code, set()).add(kind)
    rows = []
    for code, found in kinds.items():
        fired = [e for e in entries if code in e.codes()]
        suitability = code in SUITABILITY
        hit = fired if suitability else [e for e in fired if not e.suitability]
        rest = [e for e in population if code not in e.codes()]
        row = {"code": code, "kind": next(k for k in KIND_ORDER if k in found), "suitability": suitability, "fired": len(fired), **compare(hit, rest, min_sample, suitability=suitability)}
        row["by_theme"] = _by_theme(hit, rest, themes, min_sample, not suitability)
        rows.append(row)
    rows.sort(key=lambda r: (r["suitability"], KIND_ORDER.index(r["kind"]), -r["fired"], r["code"]))
    families = []
    for family in sorted({e.family for e in population if e.family}):
        mine = [e for e in population if e.family == family]
        cell = compare([e for e in mine if e.verdict == "denied"], [e for e in mine if e.verdict == "approved"], min_sample, subject=("the denied", "the approved"))
        families.append({"family": family, **group_of(mine), "denied": cell["hit"], "approved": cell["rest"], "gap": cell["gap"], "se": cell["se"], "inside_chance": cell["inside_chance"],
                         "sentence_code": cell["sentence_code"], "sentence": cell["sentence"]})
    return {"config_version": version, "reviewed": len(entries), "population": len(population), "suitability_excluded": len(entries) - len(population), "verdict": verdict, "rows": rows, "by_family": families}


def build(runs: list[ProvingRun] | None = None, *, store: ProvingStore | None = None, config: ProvingConfig | None = None, folders: list[Path] | None = None, include_track: bool = True, desks: bool = True) -> dict:
    """The report card of the Critic's rules over `runs` (every saved Time Machine run of the store, oldest first, and the Track Record, when left out). `folders` are where the research of older runs is read from (`default_folders`); `desks`
    False leaves the runs' own saved desks out of that (see `ReviewIndex`). See the
    module text for the shape: {min_sample, chance_band, limits, runs, themes, config_versions, unmatched, by_config_version: {version: {config_version, reviewed, population, suitability_excluded, verdict, rows, by_family}}}."""
    config = config or load_config()
    if runs is None:
        store = store or ProvingStore()
        runs = saved_runs(store)
        if include_track:
            from prometheus.proving import track
            runs = [*runs, track.build(store)]
    index = ReviewIndex(runs, folders if folders is not None else default_folders(store), desks)
    entries, unmatched = entries_of(runs, index)
    themes = sorted({e.theme for e in entries})
    versions = sorted({e.version for e in entries}, key=lambda v: (len(v), v))
    return {"min_sample": config.min_sample, "chance_band": float(stats_mod.CHANCE_BAND), "limits": list(LIMITS), "sentences": SENTENCES,
            "runs": [{"run_id": r.run_id, "kind": r.kind, "theme": theme_of(r), "prompt": (r.request or {}).get("prompt"), "speed": r.speed, "created_utc": r.created_utc, "reviewed": sum(1 for e in entries if e.run_id == r.run_id)} for r in runs],
            "themes": themes, "config_versions": versions, "unmatched": unmatched,
            "by_config_version": {v: _version_card(v, [e for e in entries if e.version == v], config.min_sample, themes) for v in versions}}


def saved_runs(store: ProvingStore, fixtures: Path | None = None) -> list[ProvingRun]:
    """Every saved Time Machine run of the store, oldest first, then the recorded ones of `fixtures` that the store does not have (the server's --replay mode serves them)."""
    runs: list[ProvingRun] = []
    for row in sorted(store.runs(), key=lambda r: str(r.get("created_utc"))):
        if row.get("kind") != "time_machine":
            continue
        try:
            runs.append(store.load_run(row["run_id"]))
        except (FileNotFoundError, ValueError):
            continue
    have = {r.run_id for r in runs}
    for path in sorted(fixtures.glob("*.json")) if fixtures is not None and fixtures.is_dir() else []:
        try:
            run = ProvingRun.model_validate_json(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if run.kind == "time_machine" and run.run_id not in have:
            runs.append(run)
            have.add(run.run_id)
    return sorted(runs, key=lambda r: r.created_utc)


# ---------------------------------------------------------------- as text

def _signed(value: float | None) -> str:
    return "   n/a" if value is None else f"{value * 100:+6.2f}"


def report(card: dict) -> str:
    """The card as the command prints it: for each version of the Critic, the verdict's own row (denied against approved, without the suitability denials), one line for each rule, and the Critic's verdict by theme."""
    lines = [f"CRITIC REPORT CARD  {len(card['runs'])} run(s), {len(card['themes'])} theme(s): {', '.join(card['themes'])}",
             f"  excess return over SPY in points; gap = what the rule fired on minus the others (negative: it picks losers); inside chance = within {card['chance_band']:g} standard errors; at least {card['min_sample']} with an outcome to say anything"]
    for version in card["config_versions"]:
        c = card["by_config_version"][version]
        lines += ["", f"CRITIC VERSION {version}: {c['reviewed']} reviewed recommendation(s) with an outcome, {c['suitability_excluded']} denied for the user's risk profile and left out of every comparison"]
        v = c["verdict"]
        lines.append(f"  the verdict   denied n={v['hit']['n']:<3} {_signed(v['hit']['mean_excess'])}  won {v['hit']['won']}/{v['hit']['closed']}   approved n={v['rest']['n']:<3} {_signed(v['rest']['mean_excess'])}  won {v['rest']['won']}/{v['rest']['closed']}   "
                     f"approved minus denied {_signed(None if v['gap'] is None else -v['gap'])} (se {_signed(v['se'])})")
        lines.append(f"      {v['sentence']}")
        for theme, t in v["by_theme"].items():
            lines.append(f"      {theme:<28} denied n={t['n']:<3} {_signed(t['mean_excess'])}  approved n={t['rest_n']:<3} {_signed(t['rest_mean_excess'])}   gap {_signed(t['gap'])} (se {_signed(t['se'])})  {t['sentence_code'].replace('_', ' ')}")
        lines.append(f"  {'rule':<28} {'kind':<8} {'fired':>5} {'won/closed':>10} {'hit':>7} {'rest':>7} {'gap':>7} {'se':>6}  sentence")
        for r in c["rows"]:
            lines.append(f"  {r['code']:<28} {r['kind']:<8} {r['fired']:>5} {str(r['hit']['won']) + '/' + str(r['hit']['closed']):>10} {_signed(r['hit']['mean_excess'])} {_signed(r['rest']['mean_excess'])} {_signed(r['gap'])} {_signed(r['se'])}  {r['sentence']}")
            if not r["suitability"]:
                for theme, t in r["by_theme"].items():
                    lines.append(f"      {theme:<26} n={t['n']:<3} hit {_signed(t['mean_excess'])}  rest {_signed(t['rest_mean_excess'])}  gap {_signed(t['gap'])}  {t['sentence_code'].replace('_', ' ')}")
        for f in c["by_family"]:
            lines.append(f"  family {f['family']:<15} n={f['n']:<3} {_signed(f['mean_excess'])}  denied n={f['denied']['n']:<3} {_signed(f['denied']['mean_excess'])}  approved n={f['approved']['n']:<3} {_signed(f['approved']['mean_excess'])}   {f['sentence_code'].replace('_', ' ')}")
    if card["unmatched"]:
        lines += ["", f"  {card['unmatched']} reviewed outcome(s) had no review to read the findings from and are left out"]
    lines += ["", *[f"  limit: {text}" for text in card["limits"]]]
    return "\n".join(lines)
