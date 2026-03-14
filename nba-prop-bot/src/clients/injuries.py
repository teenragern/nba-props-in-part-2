"""
Multi-source NBA injury client.

Sources (run in parallel, merged with most-pessimistic-wins logic):
  1. nba_api LeagueInjuryReport — official NBA data, typically 1-2 hr lag
  2. Rotowire injury report page — faster updates, more detail
  3. CBS Sports injury page    — fallback if Rotowire is unavailable

Consensus rule:
  Out > Doubtful > Questionable > Probable > Healthy/Unknown
  If any source calls a player Out, the merged record is Out.
"""
import time
import requests
from typing import List, Dict
from src.utils.retry import retry_with_backoff
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Severity ranking: higher = more pessimistic
_STATUS_RANK = {
    'Out': 4,
    'Doubtful': 3,
    'Questionable': 2,
    'Probable': 1,
    'Unknown': 0,
}

_HEADERS = {'User-Agent': 'Mozilla/5.0 (compatible; NBABotV2/1.0)'}


def _most_pessimistic(a: str, b: str) -> str:
    return a if _STATUS_RANK.get(a, 0) >= _STATUS_RANK.get(b, 0) else b


class InjuryClient:
    """
    Multi-source injury aggregator.
    Primary:   nba_api LeagueInjuryReport
    Secondary: Rotowire injury report page (faster, more detail)
    Tertiary:  CBS Sports (fallback if both above fail)
    """

    @retry_with_backoff(retries=3, backoff_in_seconds=2)
    def get_injuries(self) -> List[Dict[str, str]]:
        """
        Fetch and merge injury reports from all sources.
        Returns list of {player_name, team, status, description}.
        """
        nba_records   = self._fetch_nba_api()
        roto_records  = self._scrape_rotowire()

        if not nba_records and not roto_records:
            logger.warning("Primary sources failed — falling back to CBS Sports.")
            return self._scrape_cbs_sports()

        merged = self._merge_sources(nba_records, roto_records)
        logger.info(f"Injury merge complete: {len(merged)} players "
                    f"(nba_api={len(nba_records)}, rotowire={len(roto_records)})")
        return merged

    # ------------------------------------------------------------------
    # Source 1: nba_api
    # ------------------------------------------------------------------

    def _fetch_nba_api(self) -> List[Dict[str, str]]:
        try:
            from nba_api.stats.endpoints import leagueinjuryreport
            report = leagueinjuryreport.LeagueInjuryReport(league_id='00')
            time.sleep(0.6)
            df = report.get_data_frames()[0]

            injuries = []
            for _, row in df.iterrows():
                player = row.get('PLAYER_NAME', row.get('PlayerName', ''))
                team   = row.get('TEAM', row.get('Team', row.get('TEAM_NAME', '')))
                raw_st = row.get('PLAYER_STATUS', row.get('PlayerStatus',
                          row.get('CURRENT_STATUS', row.get('Status', ''))))
                desc   = row.get('RETURN_DATE', row.get('Comment', ''))
                if not player:
                    continue
                injuries.append({
                    'player_name': str(player).strip(),
                    'team':        str(team).strip(),
                    'status':      self.normalize_status(str(raw_st)),
                    'description': str(desc).strip(),
                })
            logger.info(f"nba_api: {len(injuries)} injury records.")
            return injuries
        except Exception as e:
            logger.warning(f"nba_api LeagueInjuryReport failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Source 2: Rotowire injury report
    # ------------------------------------------------------------------

    def _scrape_rotowire(self) -> List[Dict[str, str]]:
        """
        Scrape https://www.rotowire.com/basketball/injury-report.php
        Table columns: Player | Team | Pos | Injury | Status | Est. Return
        """
        try:
            from bs4 import BeautifulSoup
            resp = requests.get(
                'https://www.rotowire.com/basketball/injury-report.php',
                headers=_HEADERS,
                timeout=12,
            )
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')

            injuries = []
            # Rotowire uses a ul.injury-report__table structure
            for item in soup.select('li.injury-report__table-row'):
                try:
                    player_tag  = item.select_one('.injury-report__player-name')
                    team_tag    = item.select_one('.injury-report__team')
                    status_tag  = item.select_one('.injury-report__status')
                    injury_tag  = item.select_one('.injury-report__injury')

                    player_name = player_tag.get_text(strip=True) if player_tag else ''
                    team        = team_tag.get_text(strip=True)   if team_tag   else ''
                    raw_status  = status_tag.get_text(strip=True) if status_tag else ''
                    injury_desc = injury_tag.get_text(strip=True) if injury_tag else ''

                    if not player_name:
                        continue
                    injuries.append({
                        'player_name': player_name,
                        'team':        team,
                        'status':      self.normalize_status(raw_status),
                        'description': injury_desc,
                    })
                except Exception:
                    continue

            # Alternate selector: table-based layout
            if not injuries:
                for row in soup.select('table.injury-report tr')[1:]:
                    cols = row.select('td')
                    if len(cols) < 5:
                        continue
                    try:
                        player_name = cols[0].get_text(strip=True)
                        team        = cols[1].get_text(strip=True)
                        raw_status  = cols[4].get_text(strip=True)
                        injury_desc = cols[3].get_text(strip=True)
                        if not player_name:
                            continue
                        injuries.append({
                            'player_name': player_name,
                            'team':        team,
                            'status':      self.normalize_status(raw_status),
                            'description': injury_desc,
                        })
                    except Exception:
                        continue

            logger.info(f"Rotowire: {len(injuries)} injury records.")
            return injuries

        except Exception as e:
            logger.warning(f"Rotowire injury scraper failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Source 3: CBS Sports (fallback)
    # ------------------------------------------------------------------

    def _scrape_cbs_sports(self) -> List[Dict[str, str]]:
        try:
            from bs4 import BeautifulSoup
            resp = requests.get(
                'https://www.cbssports.com/nba/injuries/',
                headers=_HEADERS,
                timeout=10,
            )
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')

            injuries = []
            for team_section in soup.select('.TableBase'):
                header    = team_section.select_one('.TableBase-title')
                team_name = header.get_text(strip=True) if header else 'Unknown'
                for row in team_section.select('tbody tr'):
                    cols = row.select('td')
                    if len(cols) < 3:
                        continue
                    player_name = cols[0].get_text(strip=True)
                    injury_desc = cols[1].get_text(strip=True) if len(cols) > 1 else ''
                    raw_status  = cols[2].get_text(strip=True) if len(cols) > 2 else ''
                    injuries.append({
                        'player_name': player_name,
                        'team':        team_name,
                        'status':      self.normalize_status(raw_status),
                        'description': injury_desc,
                    })

            logger.info(f"CBS Sports: {len(injuries)} injury records.")
            return injuries

        except Exception as e:
            logger.error(f"CBS Sports injury scraper failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_sources(*source_lists: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Merge multiple source lists. Key = player_name (lower-cased).
        Status = most pessimistic across all sources.
        Description = concatenated (non-empty, deduplicated).
        """
        merged: Dict[str, Dict[str, str]] = {}

        for source in source_lists:
            for record in source:
                key = record['player_name'].lower().strip()
                if not key:
                    continue
                if key not in merged:
                    merged[key] = dict(record)
                else:
                    existing = merged[key]
                    existing['status'] = _most_pessimistic(existing['status'], record['status'])
                    # Append description from this source if new info
                    new_desc = record.get('description', '').strip()
                    existing_desc = existing.get('description', '').strip()
                    if new_desc and new_desc not in existing_desc:
                        existing['description'] = (
                            f"{existing_desc}; {new_desc}".strip('; ')
                        )

        return list(merged.values())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_status(raw_status: str) -> str:
        status = raw_status.lower() if raw_status else ''
        if 'out' in status:
            return 'Out'
        if 'doubtful' in status:
            return 'Doubtful'
        if 'questionable' in status or 'game time decision' in status or 'gtd' in status:
            return 'Questionable'
        if 'probable' in status:
            return 'Probable'
        return 'Unknown'
