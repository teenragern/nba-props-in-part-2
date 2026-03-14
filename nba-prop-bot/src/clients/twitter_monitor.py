"""
Real-time Twitter/X news monitor via Nitter RSS.

Polls Nitter RSS feeds for top NBA beat reporters (Shams, Woj, Underdog NBA, etc.)
every TWITTER_POLL_INTERVAL seconds. Nitter is an open-source Twitter frontend that
exposes RSS at https://{instance}/{username}/rss — no API key required.

Multiple Nitter instances are cycled in order of reliability. Falls back to
Rotowire/ESPN RSS if every Nitter instance fails on a given poll.

This class is a drop-in replacement for BreakingNewsMonitor (same interface).
"""
import feedparser
import requests
import random
from datetime import datetime
from typing import List, Dict, Set, Optional
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Beat reporter Twitter handles — ordered by signal quality.
# Shams/Woj break news first; Underdog/FantasyLabs aggregate fast.
REPORTER_HANDLES = [
    "ShamsCharania",    # The Athletic — fastest injury breaks
    "wojespn",          # ESPN — second fastest
    "UnderdogNBA",      # Underdog Fantasy — aggregates beat reporters
    "FantasyLabsNBA",   # Fantasy Labs — fast injury aggregator
    "NBAInjuries",      # Dedicated injury aggregator account
    "Josh_Robbins",     # The Athletic (East/Central beat)
    "RotowireNBA",      # Rotowire NBA desk
]

# Public Nitter instances — tried in order of fewest accumulated failures.
# If an instance 404s or errors repeatedly it gets deprioritised automatically.
NITTER_INSTANCES = [
    "nitter.privacydev.net",
    "nitter.poast.org",
    "nitter.1d4.us",
    "nitter.kavin.rocks",
    "nitter.net",
    "nitter.cz",
]

# Fallback RSS feeds used when every Nitter instance is down.
FALLBACK_FEED_URLS = [
    "https://www.rotowire.com/basketball/rss/news.xml",
    "https://www.espn.com/espn/rss/nba/news",
]

INJURY_KEYWORDS = {
    'out', 'injured', 'injury', 'doubtful', 'questionable',
    'ankle', 'knee', 'hip', 'shoulder', 'concussion', 'illness',
    'ruled out', 'day-to-day', 'dtd', 'inactive', 'scratched',
    "won't play", 'will not play', 'listed out', 'expected to miss',
    'placed on', 'left the game', 'did not play', 'dnp',
    'game-time decision', 'gtd', 'limited', 'not playing',
    'miss tonight', 'miss the game', 'probable',
}

_FETCH_TIMEOUT = 8      # seconds per Nitter request
_UA = 'Mozilla/5.0 (compatible; NBABot/1.0; +https://github.com)'


class TwitterNitterMonitor:
    """
    Polls Nitter RSS feeds for beat reporter tweets and returns new injury items.
    Drop-in replacement for BreakingNewsMonitor — same get_breaking_injuries() API.
    """

    def __init__(self):
        self._seen_ids: Set[str] = set()
        self._initialized: bool = False
        self._player_names: Set[str] = self._load_player_names()
        # Failure counts per instance — lower = preferred
        self._failures: Dict[str, int] = {inst: 0 for inst in NITTER_INSTANCES}

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    @staticmethod
    def _load_player_names() -> Set[str]:
        try:
            from nba_api.stats.static import players
            return {p['full_name'].lower() for p in players.get_players()}
        except Exception:
            return set()

    # ------------------------------------------------------------------
    # Nitter fetching
    # ------------------------------------------------------------------

    def _sorted_instances(self) -> List[str]:
        """Return instances ordered by ascending failure count."""
        return sorted(NITTER_INSTANCES, key=lambda i: self._failures[i])

    def _fetch_nitter_feed(self, handle: str) -> Optional[feedparser.FeedParserDict]:
        """Try each Nitter instance in reliability order. Return first success."""
        for instance in self._sorted_instances():
            url = f"https://{instance}/{handle}/rss"
            try:
                resp = requests.get(
                    url, timeout=_FETCH_TIMEOUT,
                    headers={'User-Agent': _UA},
                )
                if resp.status_code == 200:
                    feed = feedparser.parse(resp.content)
                    if feed.entries:
                        # Reward working instance
                        self._failures[instance] = max(0, self._failures[instance] - 1)
                        logger.debug(f"Nitter OK: {instance}/{handle} ({len(feed.entries)} entries)")
                        return feed
                    # 200 but empty feed — count as soft failure
                    self._failures[instance] += 1
                else:
                    self._failures[instance] += 1
                    logger.debug(f"Nitter {instance}/{handle} HTTP {resp.status_code}")
            except Exception as e:
                self._failures[instance] += 1
                logger.debug(f"Nitter {instance}/{handle} error: {e}")
        return None

    # ------------------------------------------------------------------
    # Fallback RSS
    # ------------------------------------------------------------------

    def _fetch_fallback_entries(self) -> List[Dict]:
        """Poll Rotowire/ESPN RSS. Returns raw dicts for _process_entry."""
        entries = []
        for url in FALLBACK_FEED_URLS:
            try:
                feed = feedparser.parse(url)
                source = feed.feed.get('title', url)
                for entry in feed.entries:
                    entries.append({
                        'id':           entry.get('id') or entry.get('link', ''),
                        'title':        entry.get('title', ''),
                        'summary':      entry.get('summary', ''),
                        'source':       source,
                        'published_at': entry.get('published', ''),
                    })
            except Exception as e:
                logger.warning(f"Fallback RSS error ({url}): {e}")
        return entries

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_injury_related(text: str) -> bool:
        text_lower = text.lower()
        return any(kw in text_lower for kw in INJURY_KEYWORDS)

    def _extract_players(self, text: str) -> List[str]:
        text_lower = text.lower()
        return [name for name in self._player_names if name in text_lower]

    def _process_entry(
        self,
        entry_id: str,
        title: str,
        summary: str,
        source: str,
        published_at: str,
    ) -> Optional[Dict]:
        """Return alert dict if new and injury-related; else None."""
        if entry_id in self._seen_ids:
            return None
        self._seen_ids.add(entry_id)

        if not self._initialized:
            return None     # first pass: just mark as seen, no alerts

        combined = title + ' ' + summary
        if not self._is_injury_related(combined):
            return None

        players_found = self._extract_players(combined)
        return {
            'player_name':  players_found[0].title() if players_found else '',
            'all_players':  [p.title() for p in players_found],
            'title':        title,
            'summary':      summary[:300],
            'source':       source,
            'published_at': published_at or datetime.now().strftime('%H:%M ET'),
        }

    # ------------------------------------------------------------------
    # Public API (same as BreakingNewsMonitor)
    # ------------------------------------------------------------------

    def get_breaking_injuries(self) -> List[Dict]:
        """
        Poll all Nitter feeds (+ fallback). Returns NEW injury items as:
            {player_name, all_players, title, summary, source, published_at}
        Returns [] on first poll (initialisation pass — marks items as seen).
        """
        new_items: List[Dict] = []
        nitter_ok = False

        for handle in REPORTER_HANDLES:
            feed = self._fetch_nitter_feed(handle)
            if feed is None:
                continue
            nitter_ok = True
            source = f"@{handle}"
            for entry in feed.entries:
                entry_id   = entry.get('id') or entry.get('link', '')
                title      = entry.get('title', '')
                # Nitter puts tweet text in both title and summary; prefer summary
                summary    = entry.get('summary', '') or title
                published  = entry.get('published', '')
                item = self._process_entry(entry_id, title, summary, source, published)
                if item:
                    new_items.append(item)

        # Fallback to Rotowire/ESPN RSS if every Nitter instance failed
        if not nitter_ok:
            logger.warning("All Nitter instances unreachable — using Rotowire/ESPN RSS fallback")
            for raw in self._fetch_fallback_entries():
                item = self._process_entry(
                    raw['id'], raw['title'], raw['summary'],
                    raw['source'], raw['published_at'],
                )
                if item:
                    new_items.append(item)

        self._initialized = True
        return new_items
