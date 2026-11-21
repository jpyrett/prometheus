"""Is it the same company? (Critic v2, `design/critic_v2_evidence_spec.md` section 3, rule C4.) A small, pure function: no model, no network, no clock.

    same_company("XOM", "Exxon Mobil", "Exxonmobil Holdings Corp")     # True
    same_company("APA", "APA Corporation", "Apa Corp")                  # True
    same_company("AMD", "AMD", "Advanced Micro Devices")                 # True: the symbol itself is named
    same_company("PG", "Procter & Gamble Co", "ACME ROCKET SKATES LTD")  # False

The record names a company (the Advisor's own words, or the Data Clerk's) and the symbol directory lists the same symbol under its own spelling. They are the same company when, with the case, the spacing, the
punctuation, the corporate words (Inc, Corp, Corporation, Holdings, Group ...) and a share-class mark (Cl A) removed, the two names are the same text; or one is the beginning of the other (the directory cuts a long name
short, "Restaurant Brands Intern", or adds a suffix, "Amazon.Com Inc" for "Amazon": at least four letters, as the Advisor's own rule asks of a name's word); or the symbol itself is a word of either name. A name that
still does not match is the `company_mismatch` finding, as it was.

THIS REPLACES a call to `advisor.names_company` (Critic v1), which is about a QUOTE naming a company: it asks for a whole word of 4 letters or more shared with the directory's name, so "Exxon Mobil" never matched
"Exxonmobil Holdings Corp" (two words against one) and "APA Corporation" never matched "Apa Corp" (the symbol only counts in capitals): in the four six-month tests of 2026-11-20 it fired 15 times and 11 of them were winners
(a bug, not a finding). The Advisor's function stays as it is for the quotes.
"""
import re
import unicodedata

# The corporate words and legal forms that name nothing in particular. A closed list (no model decides it); a company's own word that is not here is kept.
CORPORATE = frozenset({"co", "corp", "corporation", "inc", "incorporated", "ltd", "limited", "plc", "llc", "lp", "llp", "company", "holdings", "holding", "group", "sa", "nv", "ag", "se", "spa", "the"})
CLASS_MARKS = frozenset({"cl", "class"})      # "Alphabet Inc-Cl A": the mark and the one letter after it
MIN_PREFIX = 4                                 # the letters a name must have to be "the beginning of" another (the Advisor's own `names_company` asks four for a distinctive word)


def words(name: str | None) -> list[str]:
    """The words of a name, lower case, accents and apostrophes gone ("McDonald's" is "mcdonalds"), punctuation a break: ["procter", "gamble", "co"]."""
    text = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode("ascii").lower()
    return re.findall(r"[a-z0-9]+", re.sub(r"['’ʼ]", "", text))


def canonical(name: str | None) -> str:
    """A name with the corporate words and the share-class mark taken out and the spacing closed: "Exxon Mobil Corporation" and "Exxonmobil Holdings Corp" are both "exxonmobil". An empty string when nothing is left."""
    found, skip = [], False
    parts = words(name)
    for i, word in enumerate(parts):
        if skip:
            skip = False
            continue
        if word in CLASS_MARKS and i + 1 < len(parts) and len(parts[i + 1]) == 1:
            skip = True
            continue
        if word not in CORPORATE:
            found.append(word)
    return "".join(found)


def same_company(symbol: str, first: str | None, second: str | None) -> bool:
    """True when `first` and `second` are the same company's name (see the module text). False when either is missing: the caller decides what a missing name means. A symbol of one letter never counts as named (it would match a
    word of a name by accident). A name with nothing left once the corporate words are out ("Holdings Corp") is the same as no name: only the symbol can match it."""
    if not first or not second:
        return False
    a, b = canonical(first), canonical(second)
    if a and a == b:
        return True
    if len(min(a, b, key=len)) >= MIN_PREFIX and (a.startswith(b) or b.startswith(a)):
        return True
    ticker = (symbol or "").lower()
    return len(ticker) >= 2 and (ticker in words(first) or ticker in words(second))
