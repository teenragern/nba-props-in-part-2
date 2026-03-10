import time
import requests
from typing import List, Dict
from src.utils.retry import retry_with_backoff
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class InjuryClient:
    """
    Priority 1: Real injury feed.
    Primary: nba_api LeagueInjuryReport (official NBA data).
    Fallback: CBS Sports web scrape.
    """

    HEADERS = {'User-Agent': 'Mozilla/5.0 (compatible; NBABotV2/1.0)'}

    @retry_with_backoff(retries=3, backoff_in_seconds=2)
    def get_injuries(self) -> List[Dict[str, str]]:
        """Fetch injury report. Returns list of {player_name, team, status, description}."""
        logger.info("Fetching NBA injury report via nba_api LeagueInjuryReport...")
        try:
            from nba_api.stats.endpoints import leagueinjuryreport
            report = leagueinjuryreport.LeagueInjuryReport(league_id='00')
            time.sleep(0.6)
            df = report.get_data_frames()[0]

            injuries = []
            for _, row in df.iterrows():
                # Column names vary slightly across nba_api versions
                player = row.get('PLAYER_NAME', row.get('PlayerName', ''))
                team = row.get('TEAM', row.get('Team', row.get('TEAM_NAME', '')))
                raw_status = row.get('PLAYER_STATUS', row.get('PlayerStatus',
                             row.get('CURRENT_STATUS', row.get('Status', ''))))
                desc = row.get('RETURN_DATE', row.get('Comment', ''))

                if not player:
                    continue
                injuries.append({
                    'player_name': str(player).strip(),
                    'team': str(team).strip(),
                    'status': self.normalize_status(str(raw_status)),
                    'description': str(desc).strip(),
                })

            logger.info(f"Fetched {len(injuries)} injury records from nba_api.")
            return injuries

        except Exception as e:
            logger.warning(f"LeagueInjuryReport unavailable ({e}), falling back to CBS Sports scraper.")
            return self._scrape_cbs_sports()

    def _scrape_cbs_sports(self) -> List[Dict[str, str]]:
        """Fallback: scrape CBS Sports NBA injury page."""
        try:
            from bs4 import BeautifulSoup
            resp = requests.get(
                'https://www.cbssports.com/nba/injuries/',
                headers=self.HEADERS,
                timeout=10
            )
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')

            injuries = []
            for team_section in soup.select('.TableBase'):
                header = team_section.select_one('.TableBase-title')
                team_name = header.get_text(strip=True) if header else 'Unknown'

                for row in team_section.select('tbody tr'):
                    cols = row.select('td')
                    if len(cols) < 3:
                        continue
                    player_name = cols[0].get_text(strip=True)
                    injury_desc = cols[1].get_text(strip=True) if len(cols) > 1 else ''
                    raw_status = cols[2].get_text(strip=True) if len(cols) > 2 else ''
                    injuries.append({
                        'player_name': player_name,
                        'team': team_name,
                        'status': self.normalize_status(raw_status),
                        'description': injury_desc,
                    })

            logger.info(f"Scraped {len(injuries)} injury records from CBS Sports.")
            return injuries

        except Exception as e:
            logger.error(f"CBS Sports injury scraper failed: {e}")
            return []

    def normalize_status(self, raw_status: str) -> str:
        status = raw_status.lower() if raw_status else ""
        if "out" in status:
            return "Out"
        if "doubtful" in status:
            return "Doubtful"
        if "questionable" in status or "game time decision" in status or "gtd" in status:
            return "Questionable"
        if "probable" in status:
            return "Probable"
        return "Unknown"
