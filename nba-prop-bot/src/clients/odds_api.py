import requests
from typing import List, Dict, Any, Optional, Tuple
from src.config import ODDS_API_KEY, ODDS_REGION, BOOKMAKERS, SHARP_BOOKS
from src.utils.retry import retry_with_backoff
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

class OddsApiClient:
    BASE_URL = "https://api.the-odds-api.com/v4/sports"
    SPORT = "basketball_nba"

    def __init__(self, api_key: str = ODDS_API_KEY):
        self.api_key = api_key
        self.requests_used = 0
        self.requests_remaining = 0

    def _update_quota(self, headers: Any):
        used = headers.get('x-requests-used')
        remaining = headers.get('x-requests-remaining')
        if used is not None:
            self.requests_used = int(used)
        if remaining is not None:
            self.requests_remaining = int(remaining)
        logger.debug(f"Odds API Quota - Used: {self.requests_used}, Remaining: {self.requests_remaining}")

    @retry_with_backoff(retries=3, backoff_in_seconds=2)
    def get_events(self) -> List[Dict[str, Any]]:
        url = f"{self.BASE_URL}/{self.SPORT}/events"
        params = {
            "apiKey": self.api_key
        }
        logger.info("Fetching NBA events from Odds API")
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        self._update_quota(response.headers)
        return response.json()

    @retry_with_backoff(retries=3, backoff_in_seconds=2)
    def get_event_odds(self, event_id: str, markets: List[str]) -> Dict[str, Any]:
        url = f"{self.BASE_URL}/{self.SPORT}/events/{event_id}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": ODDS_REGION,
            "markets": ",".join(markets),
            "bookmakers": ",".join(list(dict.fromkeys(BOOKMAKERS + SHARP_BOOKS))) or None,
            "oddsFormat": "decimal"
        }
        params = {k: v for k, v in params.items() if v is not None}
        
        logger.info(f"Fetching odds for event {event_id} markets: {markets}")
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        self._update_quota(response.headers)
        return response.json()

    @staticmethod
    def extract_consensus_total(bookmakers: List[Dict]) -> Optional[float]:
        """
        Return the median over/under game total across all books that carry
        the totals market. Returns None when no totals data is present.
        """
        totals: List[float] = []
        for book in bookmakers:
            for mkt in book.get('markets', []):
                if mkt.get('key') != 'totals':
                    continue
                for outcome in mkt.get('outcomes', []):
                    point = outcome.get('point')
                    if point is not None:
                        totals.append(float(point))
                        break  # only need one side per book
        if not totals:
            return None
        totals.sort()
        mid = len(totals) // 2
        return (totals[mid - 1] + totals[mid]) / 2.0 if len(totals) % 2 == 0 else totals[mid]

    @staticmethod
    def extract_consensus_spread(bookmakers: List[Dict], home_team: str) -> Optional[float]:
        """
        Return the median spread for the home team across all books that carry
        the spreads market.  Convention matches the Odds API: negative = home
        team is favored (e.g. -6.5 means home gives 6.5 points).
        Returns None when no spread data is present.
        """
        spreads: List[float] = []
        home_lower = home_team.lower()
        for book in bookmakers:
            for mkt in book.get('markets', []):
                if mkt.get('key') != 'spreads':
                    continue
                for outcome in mkt.get('outcomes', []):
                    if outcome.get('name', '').lower() == home_lower:
                        point = outcome.get('point')
                        if point is not None:
                            spreads.append(float(point))
        if not spreads:
            return None
        # Median across books to filter out outliers
        spreads.sort()
        mid = len(spreads) // 2
        if len(spreads) % 2 == 0:
            return (spreads[mid - 1] + spreads[mid]) / 2.0
        return spreads[mid]

    # ── Game-market odds extraction ───────────────────────────────────────────
    # These helpers return a (price_a, price_b, book_title) tuple from the
    # sharpest available book for use in devig_shin probability estimation.

    _SHARP_PRIORITY = ['pinnacle', 'circa', 'bookmaker', 'betonlineag']

    @staticmethod
    def extract_h2h_odds(
        bookmakers: List[Dict], home_team: str, away_team: str
    ) -> Optional[Tuple[float, float, str]]:
        """
        Return (home_price, away_price, book_title) for the h2h (moneyline) market.

        Iterates bookmakers in priority order (sharpest first).  Returns the first
        book that has valid decimal prices for both sides.  Returns None when no
        h2h data is present.
        """
        home_lower = home_team.lower()
        home_last  = home_lower.split()[-1]
        away_lower = away_team.lower()
        away_last  = away_lower.split()[-1]

        candidates: Dict[str, Tuple[float, float, str]] = {}
        for book in bookmakers:
            title_lower = book.get('title', '').lower()
            for mkt in book.get('markets', []):
                if mkt.get('key') != 'h2h':
                    continue
                hp = ap = None
                for outcome in mkt.get('outcomes', []):
                    name  = outcome.get('name', '').lower()
                    price = float(outcome.get('price', 0.0))
                    if price <= 1.0:
                        continue
                    if name == home_lower or home_last in name:
                        hp = price
                    elif name == away_lower or away_last in name:
                        ap = price
                if hp and ap:
                    candidates[title_lower] = (hp, ap, book.get('title', ''))

        if not candidates:
            return None
        for sharp in OddsApiClient._SHARP_PRIORITY:
            if sharp in candidates:
                return candidates[sharp]
        return next(iter(candidates.values()))

    @staticmethod
    def extract_spread_odds_at_line(
        bookmakers: List[Dict], home_team: str, home_spread: float
    ) -> Optional[Tuple[float, float, str]]:
        """
        Return (home_price, away_price, book_title) for the spread at `home_spread`.

        `home_spread` follows the OddsApiClient convention: negative means the
        home team is favored (e.g. -6.5).  Returns None when no matching data
        is found.
        """
        home_lower = home_team.lower()
        home_last  = home_lower.split()[-1]

        candidates: Dict[str, Tuple[float, float, str]] = {}
        for book in bookmakers:
            title_lower = book.get('title', '').lower()
            for mkt in book.get('markets', []):
                if mkt.get('key') != 'spreads':
                    continue
                hp = ap = None
                for outcome in mkt.get('outcomes', []):
                    point = outcome.get('point')
                    price = float(outcome.get('price', 0.0))
                    if point is None or price <= 1.0:
                        continue
                    name    = outcome.get('name', '').lower()
                    is_home = (name == home_lower or home_last in name)
                    if is_home and abs(float(point) - home_spread) < 0.1:
                        hp = price
                    elif not is_home and abs(float(point) + home_spread) < 0.1:
                        ap = price
                if hp and ap:
                    candidates[title_lower] = (hp, ap, book.get('title', ''))

        if not candidates:
            return None
        for sharp in OddsApiClient._SHARP_PRIORITY:
            if sharp in candidates:
                return candidates[sharp]
        return next(iter(candidates.values()))

    @staticmethod
    def extract_total_odds_at_line(
        bookmakers: List[Dict], total_line: float
    ) -> Optional[Tuple[float, float, str]]:
        """
        Return (over_price, under_price, book_title) for the total at `total_line`.
        Returns None when no matching totals data is found.
        """
        candidates: Dict[str, Tuple[float, float, str]] = {}
        for book in bookmakers:
            title_lower = book.get('title', '').lower()
            for mkt in book.get('markets', []):
                if mkt.get('key') != 'totals':
                    continue
                op = up = None
                for outcome in mkt.get('outcomes', []):
                    point = outcome.get('point')
                    price = float(outcome.get('price', 0.0))
                    if point is None or price <= 1.0:
                        continue
                    if abs(float(point) - total_line) > 0.1:
                        continue
                    name = outcome.get('name', '').lower()
                    if name == 'over':
                        op = price
                    elif name == 'under':
                        up = price
                if op and up:
                    candidates[title_lower] = (op, up, book.get('title', ''))

        if not candidates:
            return None
        for sharp in OddsApiClient._SHARP_PRIORITY:
            if sharp in candidates:
                return candidates[sharp]
        return next(iter(candidates.values()))

    # ── Q1 / 1H market extractors ─────────────────────────────────────────────
    # These mirror the full-game extractors but target h2h_q1, spreads_q1,
    # and totals_q1 market keys provided by the Odds API.

    @staticmethod
    def extract_q1_h2h_odds(
        bookmakers: List[Dict], home_team: str, away_team: str
    ) -> Optional[Tuple[float, float, str]]:
        """Return (home_price, away_price, book_title) for the h2h_q1 market."""
        home_lower = home_team.lower()
        home_last  = home_lower.split()[-1]
        away_lower = away_team.lower()
        away_last  = away_lower.split()[-1]

        candidates: Dict[str, Tuple[float, float, str]] = {}
        for book in bookmakers:
            title_lower = book.get('title', '').lower()
            for mkt in book.get('markets', []):
                if mkt.get('key') != 'h2h_q1':
                    continue
                hp = ap = None
                for outcome in mkt.get('outcomes', []):
                    name  = outcome.get('name', '').lower()
                    price = float(outcome.get('price', 0.0))
                    if price <= 1.0:
                        continue
                    if name == home_lower or home_last in name:
                        hp = price
                    elif name == away_lower or away_last in name:
                        ap = price
                if hp and ap:
                    candidates[title_lower] = (hp, ap, book.get('title', ''))

        if not candidates:
            return None
        for sharp in OddsApiClient._SHARP_PRIORITY:
            if sharp in candidates:
                return candidates[sharp]
        return next(iter(candidates.values()))

    @staticmethod
    def extract_q1_spread_odds_at_line(
        bookmakers: List[Dict], home_team: str, home_q1_spread: float
    ) -> Optional[Tuple[float, float, str]]:
        """Return (home_price, away_price, book_title) for spreads_q1 at `home_q1_spread`."""
        home_lower = home_team.lower()
        home_last  = home_lower.split()[-1]

        candidates: Dict[str, Tuple[float, float, str]] = {}
        for book in bookmakers:
            title_lower = book.get('title', '').lower()
            for mkt in book.get('markets', []):
                if mkt.get('key') != 'spreads_q1':
                    continue
                hp = ap = None
                for outcome in mkt.get('outcomes', []):
                    point = outcome.get('point')
                    price = float(outcome.get('price', 0.0))
                    if point is None or price <= 1.0:
                        continue
                    name    = outcome.get('name', '').lower()
                    is_home = (name == home_lower or home_last in name)
                    if is_home and abs(float(point) - home_q1_spread) < 0.1:
                        hp = price
                    elif not is_home and abs(float(point) + home_q1_spread) < 0.1:
                        ap = price
                if hp and ap:
                    candidates[title_lower] = (hp, ap, book.get('title', ''))

        if not candidates:
            return None
        for sharp in OddsApiClient._SHARP_PRIORITY:
            if sharp in candidates:
                return candidates[sharp]
        return next(iter(candidates.values()))

    @staticmethod
    def extract_q1_total_odds_at_line(
        bookmakers: List[Dict], q1_total_line: float
    ) -> Optional[Tuple[float, float, str]]:
        """Return (over_price, under_price, book_title) for totals_q1 at `q1_total_line`."""
        candidates: Dict[str, Tuple[float, float, str]] = {}
        for book in bookmakers:
            title_lower = book.get('title', '').lower()
            for mkt in book.get('markets', []):
                if mkt.get('key') != 'totals_q1':
                    continue
                op = up = None
                for outcome in mkt.get('outcomes', []):
                    point = outcome.get('point')
                    price = float(outcome.get('price', 0.0))
                    if point is None or price <= 1.0:
                        continue
                    if abs(float(point) - q1_total_line) > 0.1:
                        continue
                    name = outcome.get('name', '').lower()
                    if name == 'over':
                        op = price
                    elif name == 'under':
                        up = price
                if op and up:
                    candidates[title_lower] = (op, up, book.get('title', ''))

        if not candidates:
            return None
        for sharp in OddsApiClient._SHARP_PRIORITY:
            if sharp in candidates:
                return candidates[sharp]
        return next(iter(candidates.values()))

    # ── 1H / First-half market extractors ────────────────────────────────────
    # Mirror the Q1 extractors but target h2h_h1, spreads_h1, totals_h1.

    @staticmethod
    def extract_h1_h2h_odds(
        bookmakers: List[Dict], home_team: str, away_team: str
    ) -> Optional[Tuple[float, float, str]]:
        """Return (home_price, away_price, book_title) for the h2h_h1 market."""
        home_lower = home_team.lower()
        home_last  = home_lower.split()[-1]
        away_lower = away_team.lower()
        away_last  = away_lower.split()[-1]

        candidates: Dict[str, Tuple[float, float, str]] = {}
        for book in bookmakers:
            title_lower = book.get('title', '').lower()
            for mkt in book.get('markets', []):
                if mkt.get('key') != 'h2h_h1':
                    continue
                hp = ap = None
                for outcome in mkt.get('outcomes', []):
                    name  = outcome.get('name', '').lower()
                    price = float(outcome.get('price', 0.0))
                    if price <= 1.0:
                        continue
                    if name == home_lower or home_last in name:
                        hp = price
                    elif name == away_lower or away_last in name:
                        ap = price
                if hp and ap:
                    candidates[title_lower] = (hp, ap, book.get('title', ''))

        if not candidates:
            return None
        for sharp in OddsApiClient._SHARP_PRIORITY:
            if sharp in candidates:
                return candidates[sharp]
        return next(iter(candidates.values()))

    @staticmethod
    def extract_h1_spread_odds_at_line(
        bookmakers: List[Dict], home_team: str, home_h1_spread: float
    ) -> Optional[Tuple[float, float, str]]:
        """Return (home_price, away_price, book_title) for spreads_h1 at `home_h1_spread`."""
        home_lower = home_team.lower()
        home_last  = home_lower.split()[-1]

        candidates: Dict[str, Tuple[float, float, str]] = {}
        for book in bookmakers:
            title_lower = book.get('title', '').lower()
            for mkt in book.get('markets', []):
                if mkt.get('key') != 'spreads_h1':
                    continue
                hp = ap = None
                for outcome in mkt.get('outcomes', []):
                    point = outcome.get('point')
                    price = float(outcome.get('price', 0.0))
                    if point is None or price <= 1.0:
                        continue
                    name    = outcome.get('name', '').lower()
                    is_home = (name == home_lower or home_last in name)
                    if is_home and abs(float(point) - home_h1_spread) < 0.1:
                        hp = price
                    elif not is_home and abs(float(point) + home_h1_spread) < 0.1:
                        ap = price
                if hp and ap:
                    candidates[title_lower] = (hp, ap, book.get('title', ''))

        if not candidates:
            return None
        for sharp in OddsApiClient._SHARP_PRIORITY:
            if sharp in candidates:
                return candidates[sharp]
        return next(iter(candidates.values()))

    @staticmethod
    def extract_h1_total_odds_at_line(
        bookmakers: List[Dict], h1_total_line: float
    ) -> Optional[Tuple[float, float, str]]:
        """Return (over_price, under_price, book_title) for totals_h1 at `h1_total_line`."""
        candidates: Dict[str, Tuple[float, float, str]] = {}
        for book in bookmakers:
            title_lower = book.get('title', '').lower()
            for mkt in book.get('markets', []):
                if mkt.get('key') != 'totals_h1':
                    continue
                op = up = None
                for outcome in mkt.get('outcomes', []):
                    point = outcome.get('point')
                    price = float(outcome.get('price', 0.0))
                    if point is None or price <= 1.0:
                        continue
                    if abs(float(point) - h1_total_line) > 0.1:
                        continue
                    name = outcome.get('name', '').lower()
                    if name == 'over':
                        op = price
                    elif name == 'under':
                        up = price
                if op and up:
                    candidates[title_lower] = (op, up, book.get('title', ''))

        if not candidates:
            return None
        for sharp in OddsApiClient._SHARP_PRIORITY:
            if sharp in candidates:
                return candidates[sharp]
        return next(iter(candidates.values()))

    # ── Team totals extractors ─────────────────────────────────────────────────
    # The team_totals market has outcomes with name="Over"/"Under" and
    # description=team_name.  Each team gets its own Over/Under pair.

    @staticmethod
    def extract_consensus_team_total(
        bookmakers: List[Dict], team_name: str
    ) -> Optional[float]:
        """
        Return the median team total line across all books that carry the
        team_totals market for `team_name`.  Used as the line anchor.
        """
        team_lower = team_name.lower()
        team_last  = team_lower.split()[-1]
        totals: List[float] = []
        for book in bookmakers:
            for mkt in book.get('markets', []):
                if mkt.get('key') != 'team_totals':
                    continue
                for outcome in mkt.get('outcomes', []):
                    desc  = outcome.get('description', '').lower()
                    point = outcome.get('point')
                    if point is None:
                        continue
                    if desc == team_lower or team_last in desc:
                        totals.append(float(point))
                        break  # one line per team per book
        if not totals:
            return None
        totals.sort()
        mid = len(totals) // 2
        return (totals[mid - 1] + totals[mid]) / 2.0 if len(totals) % 2 == 0 else totals[mid]

    @staticmethod
    def extract_team_total_odds(
        bookmakers: List[Dict], team_name: str, team_total_line: float
    ) -> Optional[Tuple[float, float, str]]:
        """
        Return (over_price, under_price, book_title) for the team_totals market
        at `team_total_line` for `team_name`.
        """
        team_lower = team_name.lower()
        team_last  = team_lower.split()[-1]

        candidates: Dict[str, Tuple[float, float, str]] = {}
        for book in bookmakers:
            title_lower = book.get('title', '').lower()
            for mkt in book.get('markets', []):
                if mkt.get('key') != 'team_totals':
                    continue
                op = up = None
                for outcome in mkt.get('outcomes', []):
                    desc  = outcome.get('description', '').lower()
                    point = outcome.get('point')
                    price = float(outcome.get('price', 0.0))
                    if point is None or price <= 1.0:
                        continue
                    if abs(float(point) - team_total_line) > 0.1:
                        continue
                    if desc != team_lower and team_last not in desc:
                        continue
                    name = outcome.get('name', '').lower()
                    if name == 'over':
                        op = price
                    elif name == 'under':
                        up = price
                if op and up:
                    candidates[title_lower] = (op, up, book.get('title', ''))

        if not candidates:
            return None
        for sharp in OddsApiClient._SHARP_PRIORITY:
            if sharp in candidates:
                return candidates[sharp]
        return next(iter(candidates.values()))
