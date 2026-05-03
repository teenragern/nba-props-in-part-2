"""
Web-scraped NBA injury fallback.

Primary feed is BDL (`bdl_bridge.get_injuries_for_date`); this client provides
a secondary independent source so a single feed going stale can't silently
zero-out the OUT list.

Only working source today is CBS Sports. The previous nba_api endpoint
(`leagueinjuryreport`) was removed upstream, and the Rotowire injury page is
now a JS-rendered SPA with no parseable HTML — both have been retired.
"""
import requests
from typing import List, Dict
from src.utils.retry import retry_with_backoff
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Severity ranking — exported for sync_injuries' merge logic
_STATUS_RANK = {
    'Out':          4,
    'Doubtful':     3,
    'Questionable': 2,
    'Probable':     1,
    'Unknown':      0,
}

_HEADERS = {'User-Agent': 'Mozilla/5.0 (compatible; NBABotV2/1.0)'}


class InjuryClient:
    """Secondary injury source. Today: CBS Sports only."""

    @retry_with_backoff(retries=3, backoff_in_seconds=2)
    def get_injuries(self) -> List[Dict[str, str]]:
        """Returns list of {player_name, team, status, description, return_date}."""
        return self._scrape_cbs_sports()

    # ------------------------------------------------------------------
    # CBS Sports
    # ------------------------------------------------------------------
    #
    # Page layout (verified 2026-04):
    #   .TableBase                       — one section per team
    #     .TableBase-title               — team name
    #     tbody tr.TableBase-bodyTr      — one row per injured player
    #       td[0]  .CellPlayerName--long → full player name
    #       td[1]  position
    #       td[2]  .CellGameDate          → "Wed, Apr 22" (last update)
    #       td[3]  injury body part       → "Abdomen"
    #       td[4]  status / return ETA    → "Expected to be out until at least Jul 2"

    def _scrape_cbs_sports(self) -> List[Dict[str, str]]:
        try:
            from bs4 import BeautifulSoup
            resp = requests.get(
                'https://www.cbssports.com/nba/injuries/',
                headers=_HEADERS,
                timeout=12,
            )
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')

            injuries: List[Dict[str, str]] = []
            for section in soup.select('.TableBase'):
                header = section.select_one('.TableBase-title')
                team = header.get_text(strip=True) if header else 'Unknown'

                for row in section.select('tbody tr.TableBase-bodyTr'):
                    cols = row.select('td')
                    if len(cols) < 5:
                        continue

                    name_tag = cols[0].select_one('.CellPlayerName--long a') \
                               or cols[0].select_one('.CellPlayerName--long') \
                               or cols[0].select_one('a')
                    player_name = name_tag.get_text(strip=True) if name_tag else cols[0].get_text(strip=True)
                    if not player_name:
                        continue

                    description = cols[3].get_text(strip=True)
                    status_text = cols[4].get_text(' ', strip=True)
                    return_date = self._extract_return_date(status_text)

                    injuries.append({
                        'player_name': player_name,
                        'team':        team,
                        'status':      self.normalize_status(status_text),
                        'description': description,
                        'return_date': return_date,
                    })

            logger.info(f"CBS Sports: {len(injuries)} injury records.")
            return injuries

        except Exception as e:
            logger.error(f"CBS Sports injury scraper failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_return_date(status_text: str) -> str:
        """Pull a 'Mon DD' return date from CBS status blurbs like
        'Expected to be out until at least Jul 2' → 'Jul 2'."""
        if not status_text:
            return ''
        import re
        m = re.search(
            r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}',
            status_text,
        )
        return m.group(0) if m else ''

    @staticmethod
    def normalize_status(raw_status: str) -> str:
        s = (raw_status or '').lower()
        # Order matters — check specific tokens before generic 'out'
        if 'game time decision' in s or 'gtd' in s:
            return 'Questionable'
        if 'day-to-day' in s or 'day to day' in s or 'dtd' in s:
            return 'Questionable'
        if 'doubtful' in s:
            return 'Doubtful'
        if 'questionable' in s:
            return 'Questionable'
        if 'probable' in s:
            return 'Probable'
        if 'out' in s or 'expected to be out' in s or 'ruled out' in s:
            return 'Out'
        return 'Unknown'
