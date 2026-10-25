"""NewsTest: run DataAgent.fetch_news, discover and fetch_press_releases, fetch_filings and search (independent of the historical download).

Run:  ~/miniforge3/envs/bmd/bin/python test_news.py
Edit CASES / FROM_DAYS below to change what is fetched. Output goes to NEWS_OUTPUT_DIR (.env) or ./news.
"""
import sys

from data_agent import DataAgent

FROM_DAYS = 7
CASES = [
    {"company": "Apple", "symbol": "AAPL", "keywords": ["iPhone", "earnings"]},
    {"company": "NVIDIA", "symbol": "NVDA", "keywords": []},
    {"company": "Tesla", "symbol": "TSLA", "keywords": ["Cybertruck", "robotaxi"]},
]
PHRASES = ["weight loss", "GLP-1"]
PRESS_RELEASE_SYMBOLS = ["AAPL"]
PRESS_RELEASE_DAYS = 365
FILING_SYMBOLS = ["AAPL"]
FILING_DAYS = 365
SEARCHES = [{"symbol": "META", "terms": ["Zuckerberg", "wealth tax", "billionaire tax"]}]
SEARCH_DAYS = 7


class NewsTest:
    def __init__(self, cases: list[dict] | None = None, phrases: list[str] | None = None,
                 from_days: int = FROM_DAYS, press_release_symbols: list[str] | None = None,
                 press_release_days: int = PRESS_RELEASE_DAYS, filing_symbols: list[str] | None = None,
                 filing_days: int = FILING_DAYS, searches: list[dict] | None = None,
                 search_days: int = SEARCH_DAYS):
        self.agent = DataAgent()
        self.searches = SEARCHES if searches is None else searches
        self.search_days = search_days
        self.filing_symbols = FILING_SYMBOLS if filing_symbols is None else filing_symbols
        self.filing_days = filing_days
        self.press_release_symbols = PRESS_RELEASE_SYMBOLS if press_release_symbols is None else press_release_symbols
        self.press_release_days = press_release_days
        self.cases = CASES if cases is None else cases
        self.phrases = PHRASES if phrases is None else phrases
        self.from_days = from_days

    def run(self) -> int:
        """Fetch news for every case. Stops at the first error and returns 1."""
        for case in self.cases:
            try:
                stats = self.agent.fetch_news(
                    company=case.get("company"), symbol=case["symbol"],
                    keywords=case.get("keywords"), from_date=self.from_days,
                )
            except Exception as e:
                print(f"\nERROR ({case['symbol']}): {e}", file=sys.stderr)
                return 1
            print(f"  {stats}\n")
        for phrase in self.phrases:
            try:
                self.agent.discover(phrase, from_date=self.from_days)
            except Exception as e:
                print(f"\nERROR (discover '{phrase}'): {e}", file=sys.stderr)
                return 1
            print()
        for symbol in self.press_release_symbols:
            try:
                self.agent.fetch_press_releases(symbol, from_date=self.press_release_days)
            except Exception as e:
                print(f"\nERROR (press releases {symbol}): {e}", file=sys.stderr)
                return 1
            print()
        for symbol in self.filing_symbols:
            try:
                self.agent.fetch_filings(symbol, from_date=self.filing_days)
            except Exception as e:
                print(f"\nERROR (filings {symbol}): {e}", file=sys.stderr)
                return 1
            print()
        try:
            self.agent.sync_cnbc_news()
        except Exception as e:
            print(f"\nERROR (CNBC RSS): {e}", file=sys.stderr)
            return 1
        print()
        for s in self.searches:
            try:
                self.agent.search(s["symbol"], s["terms"], from_date=self.search_days)
            except Exception as e:
                print(f"\nERROR (search {s['symbol']}): {e}", file=sys.stderr)
                return 1
            print()
        print("Done.")
        return 0


if __name__ == "__main__":
    sys.exit(NewsTest().run())
