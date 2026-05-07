"""
Playwright scraper microservice — Phase 3B.

Runs as a standalone FastAPI application in a separate Railway service
(using the Playwright Docker image which includes browser binaries).

The main bot's PlaywrightScraperClient calls this service over HTTP.

Endpoints:
    GET /health                        → {"status": "ok"}
    GET /scrape/props                  → list of prop lines for a player + book
    GET /scrape/game                   → list of game-market lines for a matchup

Query params for /scrape/props:
    player    Full player name, e.g. "LeBron James"
    book      Book slug: "draftkings" | "fanduel" | "betmgm" | "caesars"
    markets   Optional comma-separated market list, e.g. "player_points,player_rebounds"

Query params for /scrape/game:
    home      Home team name or abbreviation
    away      Away team name or abbreviation
    book      Book slug

Environment:
    SCRAPER_PROXY_URL     Optional HTTP/HTTPS proxy for residential IP rotation
    BROWSER_HEADLESS      "true" (default) / "false"
    SCRAPER_PAGE_TIMEOUT  Page timeout ms (default 20000)

Usage (local):
    pip install fastapi uvicorn playwright
    playwright install chromium
    uvicorn src.workers.scraper_worker:app --host 0.0.0.0 --port 8080

Deploy:
    See deploy/scraper.Dockerfile
"""

import os
import re
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

app = FastAPI(title="NBA Props Scraper Worker", version="1.0.0")

_PROXY_URL       = os.getenv("SCRAPER_PROXY_URL", "")
_HEADLESS        = os.getenv("BROWSER_HEADLESS", "true").lower() != "false"
_PAGE_TIMEOUT_MS = int(os.getenv("SCRAPER_PAGE_TIMEOUT", "20000"))

# Book URL templates — only soft books worth scraping (sharp books via Odds API)
_BOOK_URLS: Dict[str, str] = {
    "draftkings": "https://sportsbook.draftkings.com/nba-player-props",
    "fanduel":    "https://sportsbook.fanduel.com/basketball/nba",
    "betmgm":     "https://sports.betmgm.com/en/sports/basketball-7/betting/usa/nba-6004",
    "caesars":    "https://www.caesars.com/sportsbook-and-casino/sport/basketball/nba",
}

_MARKET_PATTERNS = {
    "player_points":   re.compile(r"(\d+\.?\d*)\s+(?:points?|pts?)", re.IGNORECASE),
    "player_rebounds": re.compile(r"(\d+\.?\d*)\s+(?:rebounds?|reb)", re.IGNORECASE),
    "player_assists":  re.compile(r"(\d+\.?\d*)\s+(?:assists?|ast)", re.IGNORECASE),
    "player_threes":   re.compile(r"(\d+\.?\d*)\s+(?:threes?|3-pointers?|3pm)", re.IGNORECASE),
}

_DEFAULT_MARKETS = list(_MARKET_PATTERNS.keys())


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/scrape/props")
async def scrape_props(
    player: str = Query(..., description="Full player name"),
    book:   str = Query(..., description="Book slug, e.g. draftkings"),
    markets: Optional[str] = Query(None, description="Comma-separated market list"),
) -> JSONResponse:
    if book not in _BOOK_URLS:
        raise HTTPException(status_code=400, detail=f"Unknown book '{book}'. "
                            f"Valid: {list(_BOOK_URLS.keys())}")

    target_markets = markets.split(",") if markets else _DEFAULT_MARKETS
    target_markets = [m.strip() for m in target_markets if m.strip()]

    try:
        results = await _scrape_player_props(book, player, target_markets)
        return JSONResponse(content=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/scrape/game")
async def scrape_game(
    home: str = Query(..., description="Home team name"),
    away: str = Query(..., description="Away team name"),
    book: str = Query(..., description="Book slug"),
) -> JSONResponse:
    if book not in _BOOK_URLS:
        raise HTTPException(status_code=400, detail=f"Unknown book '{book}'.")

    try:
        results = await _scrape_game_markets(book, home, away)
        return JSONResponse(content=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Scraping logic ────────────────────────────────────────────────────────────

async def _launch_browser():
    """Create a Playwright browser instance with optional proxy."""
    from playwright.async_api import async_playwright

    pw = await async_playwright().start()
    launch_kwargs: Dict[str, Any] = {"headless": _HEADLESS}
    if _PROXY_URL:
        launch_kwargs["proxy"] = {"server": _PROXY_URL}

    browser = await pw.chromium.launch(**launch_kwargs)
    return pw, browser


async def _scrape_player_props(
    book: str,
    player_name: str,
    markets: List[str],
) -> List[Dict[str, Any]]:
    """
    Navigate to the book's prop page, search for the player, and parse lines.

    Returns list of:
        {"market": str, "line": float, "over_odds": float, "under_odds": float}
    """
    pw, browser = await _launch_browser()
    results: List[Dict[str, Any]] = []

    try:
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        )
        page = await context.new_page()
        page.set_default_timeout(_PAGE_TIMEOUT_MS)

        url = _BOOK_URLS[book]
        await page.goto(url, wait_until="domcontentloaded")

        # Generic text extraction: grab all text blobs from the page that contain
        # the player name, then apply market regex patterns.
        # Each book has unique DOM structure — extend per-book selectors as needed.
        content = await page.content()
        results = _parse_props_from_html(content, player_name, markets, book)

    finally:
        await browser.close()
        await pw.stop()

    return results


async def _scrape_game_markets(
    book: str,
    home: str,
    away: str,
) -> List[Dict[str, Any]]:
    """
    Scrape spread / total / ML for a specific matchup.
    Returns list of {"market": str, "line": float, "over_odds": float, "under_odds": float}.
    """
    pw, browser = await _launch_browser()
    results: List[Dict[str, Any]] = []

    try:
        context = await browser.new_context()
        page = await context.new_page()
        page.set_default_timeout(_PAGE_TIMEOUT_MS)

        url = _BOOK_URLS[book]
        await page.goto(url, wait_until="domcontentloaded")

        content = await page.content()
        results = _parse_game_from_html(content, home, away)

    finally:
        await browser.close()
        await pw.stop()

    return results


def _parse_props_from_html(
    html: str,
    player_name: str,
    markets: List[str],
    book: str,
) -> List[Dict[str, Any]]:
    """
    Generic HTML prop parser using regex patterns.
    Override per book for more precise selectors as coverage grows.
    """
    results: List[Dict[str, Any]] = []
    name_lower = player_name.lower()

    # Find all text chunks near the player name (±2000 chars window)
    for match in re.finditer(re.escape(name_lower), html.lower()):
        start = max(0, match.start() - 500)
        end   = min(len(html), match.end() + 2000)
        chunk = html[start:end]

        for market in markets:
            pattern = _MARKET_PATTERNS.get(market)
            if pattern is None:
                continue
            m = pattern.search(chunk)
            if not m:
                continue

            line = float(m.group(1))
            # Attempt to extract American odds near the line
            over_odds, under_odds = _extract_odds_near(chunk, m.start())

            results.append({
                "market":     market,
                "line":       line,
                "over_odds":  over_odds,
                "under_odds": under_odds,
            })
            break  # one result per market per player

    return results


def _parse_game_from_html(html: str, home: str, away: str) -> List[Dict[str, Any]]:
    """Stub game-market parser — extend per book as needed."""
    results: List[Dict[str, Any]] = []
    # Look for total (over/under) line near team names
    query = f"{home.lower()}.*?{away.lower()}"
    m = re.search(query, html.lower())
    if not m:
        return results

    chunk = html[m.start():m.start() + 3000]
    total_m = re.search(r"(?:total|o/u)\s*(\d+\.?\d*)", chunk, re.IGNORECASE)
    if total_m:
        line = float(total_m.group(1))
        over_o, under_o = _extract_odds_near(chunk, total_m.start())
        results.append({
            "market":     "totals",
            "line":       line,
            "over_odds":  over_o,
            "under_odds": under_o,
        })

    return results


def _extract_odds_near(text: str, pos: int, window: int = 300) -> tuple:
    """
    Try to extract American odds (e.g. -110, +120) near position `pos`.
    Returns (over_odds, under_odds) as floats, defaulting to -110.
    """
    chunk = text[max(0, pos - 50): pos + window]
    odds  = re.findall(r"([+-]\d{3,4})", chunk)
    if len(odds) >= 2:
        return float(odds[0]), float(odds[1])
    if len(odds) == 1:
        return float(odds[0]), -110.0
    return -110.0, -110.0
