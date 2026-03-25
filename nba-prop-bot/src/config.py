import os
from dotenv import load_dotenv

load_dotenv()

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
EDGE_MIN = float(os.getenv("EDGE_MIN", "0.05"))
MIN_PROJECTED_MINUTES = float(os.getenv("MIN_PROJECTED_MINUTES", "15.0"))
ODDS_REGION = os.getenv("ODDS_REGION", "us")
BOOKMAKERS_RAW = os.getenv("BOOKMAKERS", "draftkings,fanduel,betmgm,caesars")
BOOKMAKERS = BOOKMAKERS_RAW.split(",") if BOOKMAKERS_RAW else []

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
DB_PATH = os.getenv("DB_PATH", "props.db")

# Run Risk Management
BANKROLL = float(os.getenv("BANKROLL", "1000.0"))
KELLY_FRACTION = float(os.getenv("KELLY_FRACTION", "0.25"))

# Scheduler credit-conservation settings
SCAN_INTERVAL_MINUTES = int(os.getenv("SCAN_INTERVAL_MINUTES", "90"))
QUOTA_FLOOR = int(os.getenv("QUOTA_FLOOR", "30"))
NEWS_POLL_INTERVAL     = int(os.getenv("NEWS_POLL_INTERVAL",     "60"))
TWITTER_POLL_INTERVAL  = int(os.getenv("TWITTER_POLL_INTERVAL",  "15"))

# Sharp-line devigging — top-down signal
SHARP_BOOKS_RAW = os.getenv("SHARP_BOOKS", "pinnacle")
SHARP_BOOKS     = [b.strip() for b in SHARP_BOOKS_RAW.split(",") if b.strip()]
REC_BOOKS_RAW   = os.getenv("REC_BOOKS", "draftkings,fanduel")
REC_BOOKS       = [b.strip() for b in REC_BOOKS_RAW.split(",") if b.strip()]
SHARP_EDGE_MIN  = float(os.getenv("SHARP_EDGE_MIN", "0.03"))

# Consensus sharp books for synthetic true probability
CONSENSUS_BOOKS_RAW = os.getenv("CONSENSUS_BOOKS", "pinnacle,circa,bookmaker")
CONSENSUS_BOOKS     = [b.strip().lower() for b in CONSENSUS_BOOKS_RAW.split(",") if b.strip()]
CONSENSUS_HOLD_MAX  = float(os.getenv("CONSENSUS_HOLD_MAX", "0.06"))

# ── BallDontLie GOAT Tier ─────────────────────────────────────────────
BDL_API_KEY = os.getenv("BDL_API_KEY", "")
# Use BDL as primary data source for props, injuries, lineups
# Falls back to Odds API + scraping when BDL_API_KEY is empty
BDL_ENABLED = bool(BDL_API_KEY)
# BDL vendors to pull prop lines from (rec books — sharp pricing still from Odds API)
BDL_PROP_VENDORS_RAW = os.getenv("BDL_PROP_VENDORS", "draftkings,fanduel,caesars,betmgm")
BDL_PROP_VENDORS = [v.strip() for v in BDL_PROP_VENDORS_RAW.split(",") if v.strip()]
# When BDL is active, reduce Odds API scan frequency (only needed for Pinnacle)
BDL_SHARP_SCAN_INTERVAL = int(os.getenv("BDL_SHARP_SCAN_INTERVAL", "30"))  # minutes

PROP_MARKETS = [
    "player_points",
    "player_rebounds",
    "player_assists",
    "player_threes",
    "player_points_rebounds_assists",
    "player_blocks",
    "player_steals",
]
