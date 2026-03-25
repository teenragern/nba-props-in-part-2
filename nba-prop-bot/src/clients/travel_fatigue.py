"""
Travel Fatigue Model.

Not all back-to-backs are equal:
  • Home B2B (same arena) — minimal fatigue.
  • LA → Denver (altitude) — severe: -8% minutes + altitude penalty.
  • Pacific → Eastern on B2B — 3-hour clock shift, late arrival, -7% minutes.

fatigue_multiplier is applied to projected_minutes in projections.py.
The three raw features (miles_traveled, tz_shift_hours, altitude_flag) are
also added to the ML feature vector so XGBoost can learn their interaction
with each specific market.

Penalty formula (B2B only):
  penalty = (miles / 1000) × 0.02          # 2% per 1 000 miles
           + max(0, tz_shift) × 0.015       # 1.5% per hour east
           + 0.04 × altitude_flag           # 4% flat for Denver / Utah
  fatigue_multiplier = max(0.88, 1.0 - penalty)

Non-B2B altitude (Denver/Utah regular game): 0.97 multiplier.
"""

import math
from typing import Dict, Any, Optional, Tuple

import pandas as pd


# ── Arena database ──────────────────────────────────────────────────────────
# elev_ft:   feet above sea level
# tz_offset: hours from Eastern Time  (ET=0, CT=-1, MT=-2, PT=-3)
ARENAS: Dict[str, Dict] = {
    'ATL': {'lat': 33.7573,  'lon':  -84.3963,  'elev_ft':  1050, 'tz_offset':  0},
    'BOS': {'lat': 42.3662,  'lon':  -71.0621,  'elev_ft':    20, 'tz_offset':  0},
    'BKN': {'lat': 40.6826,  'lon':  -73.9754,  'elev_ft':    20, 'tz_offset':  0},
    'CHA': {'lat': 35.2251,  'lon':  -80.8392,  'elev_ft':   751, 'tz_offset':  0},
    'CHI': {'lat': 41.8807,  'lon':  -87.6742,  'elev_ft':   596, 'tz_offset': -1},
    'CLE': {'lat': 41.4965,  'lon':  -81.6882,  'elev_ft':   654, 'tz_offset':  0},
    'DAL': {'lat': 32.7905,  'lon':  -96.8103,  'elev_ft':   430, 'tz_offset': -1},
    'DEN': {'lat': 39.7487,  'lon': -105.0077,  'elev_ft':  5183, 'tz_offset': -2},
    'DET': {'lat': 42.3410,  'lon':  -83.0551,  'elev_ft':   600, 'tz_offset':  0},
    'GSW': {'lat': 37.7680,  'lon': -122.3877,  'elev_ft':    20, 'tz_offset': -3},
    'HOU': {'lat': 29.7508,  'lon':  -95.3621,  'elev_ft':    43, 'tz_offset': -1},
    'IND': {'lat': 39.7638,  'lon':  -86.1555,  'elev_ft':   717, 'tz_offset':  0},
    'LAC': {'lat': 33.8958,  'lon': -118.3392,  'elev_ft':   102, 'tz_offset': -3},
    'LAL': {'lat': 34.0430,  'lon': -118.2673,  'elev_ft':   243, 'tz_offset': -3},
    'MEM': {'lat': 35.1383,  'lon':  -90.0505,  'elev_ft':   284, 'tz_offset': -1},
    'MIA': {'lat': 25.7814,  'lon':  -80.1870,  'elev_ft':    10, 'tz_offset':  0},
    'MIL': {'lat': 43.0451,  'lon':  -87.9170,  'elev_ft':   634, 'tz_offset': -1},
    'MIN': {'lat': 44.9795,  'lon':  -93.2762,  'elev_ft':   841, 'tz_offset': -1},
    'NOP': {'lat': 29.9490,  'lon':  -90.0821,  'elev_ft':     3, 'tz_offset': -1},
    'NYK': {'lat': 40.7505,  'lon':  -73.9934,  'elev_ft':    33, 'tz_offset':  0},
    'OKC': {'lat': 35.4634,  'lon':  -97.5151,  'elev_ft':  1197, 'tz_offset': -1},
    'ORL': {'lat': 28.5392,  'lon':  -81.3837,  'elev_ft':    96, 'tz_offset':  0},
    'PHI': {'lat': 39.9012,  'lon':  -75.1720,  'elev_ft':    18, 'tz_offset':  0},
    'PHX': {'lat': 33.4457,  'lon': -112.0712,  'elev_ft':  1086, 'tz_offset': -2},
    'POR': {'lat': 45.5316,  'lon': -122.6668,  'elev_ft':    40, 'tz_offset': -3},
    'SAC': {'lat': 38.5805,  'lon': -121.4993,  'elev_ft':    30, 'tz_offset': -3},
    'SAS': {'lat': 29.4271,  'lon':  -98.4375,  'elev_ft':   659, 'tz_offset': -1},
    'TOR': {'lat': 43.6435,  'lon':  -79.3791,  'elev_ft':   243, 'tz_offset':  0},
    'UTA': {'lat': 40.7683,  'lon': -111.9011,  'elev_ft':  4327, 'tz_offset': -2},
    'WAS': {'lat': 38.8981,  'lon':  -77.0209,  'elev_ft':    25, 'tz_offset':  0},
}

# Full team name → abbreviation (nba_api full names → ARENAS keys)
TEAM_NAME_TO_ABBR: Dict[str, str] = {
    'Atlanta Hawks':           'ATL',
    'Boston Celtics':          'BOS',
    'Brooklyn Nets':           'BKN',
    'Charlotte Hornets':       'CHA',
    'Chicago Bulls':           'CHI',
    'Cleveland Cavaliers':     'CLE',
    'Dallas Mavericks':        'DAL',
    'Denver Nuggets':          'DEN',
    'Detroit Pistons':         'DET',
    'Golden State Warriors':   'GSW',
    'Houston Rockets':         'HOU',
    'Indiana Pacers':          'IND',
    'Los Angeles Clippers':    'LAC',
    'Los Angeles Lakers':      'LAL',
    'Memphis Grizzlies':       'MEM',
    'Miami Heat':              'MIA',
    'Milwaukee Bucks':         'MIL',
    'Minnesota Timberwolves':  'MIN',
    'New Orleans Pelicans':    'NOP',
    'New York Knicks':         'NYK',
    'Oklahoma City Thunder':   'OKC',
    'Orlando Magic':           'ORL',
    'Philadelphia 76ers':      'PHI',
    'Phoenix Suns':            'PHX',
    'Portland Trail Blazers':  'POR',
    'Sacramento Kings':        'SAC',
    'San Antonio Spurs':       'SAS',
    'Toronto Raptors':         'TOR',
    'Utah Jazz':               'UTA',
    'Washington Wizards':      'WAS',
}

_HIGH_ALTITUDE_FT = 4000   # Denver (5 183 ft) and Utah (4 327 ft) exceed this

_NO_FATIGUE: Dict[str, Any] = {
    'fatigue_multiplier': 1.0,
    'miles_traveled':     0.0,
    'tz_shift_hours':     0,
    'altitude_flag':      False,
}


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two GPS coordinates, in miles."""
    R = 3958.8
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(min(1.0, a)))


def arena_from_matchup(matchup: str, own_abbr: str) -> Optional[str]:
    """
    Return the abbreviation of the arena where the game was played.

    nba_api MATCHUP format:
      "TEAM vs. OPP"  →  home game  →  own arena (own_abbr)
      "TEAM @ OPP"    →  away game  →  opponent's arena
    """
    if ' vs. ' in matchup:
        return own_abbr.upper()
    if ' @ ' in matchup:
        return matchup.split(' @ ')[-1].strip().upper()
    return None


def travel_features_for_game(
    curr_matchup: str,
    prev_matchup: str,
    own_abbr: str,
) -> Tuple[float, int, bool]:
    """
    Compute the three raw travel ML features for a single game.

    Used by ml_model.build_training_data() for each historical training row.

    Parameters
    ----------
    curr_matchup : MATCHUP string for the game being featurized.
    prev_matchup : MATCHUP string for the immediately preceding game.
    own_abbr     : The player's team abbreviation.

    Returns
    -------
    (miles_traveled, tz_shift_hours, altitude_flag)
    """
    curr_abbr = arena_from_matchup(curr_matchup, own_abbr)
    prev_abbr = arena_from_matchup(prev_matchup, own_abbr)

    curr_arena = ARENAS.get(curr_abbr) if curr_abbr else None
    prev_arena = ARENAS.get(prev_abbr) if prev_abbr else None

    alt = bool(curr_arena and curr_arena['elev_ft'] >= _HIGH_ALTITUDE_FT)

    if curr_arena and prev_arena:
        miles    = haversine_miles(
            prev_arena['lat'], prev_arena['lon'],
            curr_arena['lat'], curr_arena['lon'],
        )
        tz_shift = curr_arena['tz_offset'] - prev_arena['tz_offset']
    else:
        miles, tz_shift = 0.0, 0

    return miles, tz_shift, alt


def compute_travel_fatigue(
    player_team_abbr: str,
    today_arena_abbr: str,
    logs: pd.DataFrame,
    b2b_flag: bool = False,
) -> Dict[str, Any]:
    """
    Compute travel-adjusted fatigue for a player's upcoming game.

    Parameters
    ----------
    player_team_abbr : The player's own team abbreviation (e.g. 'LAL').
                       Used to resolve home vs. away in the most recent log game.
    today_arena_abbr : Arena abbreviation for tonight's game (always the home
                       team's abbreviation — home games are played at home).
    logs             : Player game log DataFrame (nba_api format, newest-first).
    b2b_flag         : True when today is the second night of a back-to-back.

    Returns
    -------
    dict:
        fatigue_multiplier  float [0.88, 1.0]
        miles_traveled      float straight-line miles since last game
        tz_shift_hours      int   hours east (+) / west (-) vs. last arena
        altitude_flag       bool  True if tonight's venue ≥ 4 000 ft
    """
    today_arena = ARENAS.get(today_arena_abbr.upper())
    if today_arena is None:
        return dict(_NO_FATIGUE)

    altitude = today_arena['elev_ft'] >= _HIGH_ALTITUDE_FT

    if not b2b_flag:
        if altitude:
            return {'fatigue_multiplier': 0.97, 'miles_traveled': 0.0,
                    'tz_shift_hours': 0, 'altitude_flag': True}
        return dict(_NO_FATIGUE)

    # ── B2B: figure out yesterday's location from the most recent log game ──
    prev_arena_abbr: Optional[str] = None
    if not logs.empty and 'MATCHUP' in logs.columns:
        own = (player_team_abbr or '').upper()
        prev_arena_abbr = arena_from_matchup(str(logs.iloc[0].get('MATCHUP', '')), own)

    prev_arena = ARENAS.get(prev_arena_abbr) if prev_arena_abbr else None

    if prev_arena is None:
        mult = 0.97 - (0.04 if altitude else 0.0)
        return {'fatigue_multiplier': round(max(0.88, mult), 4),
                'miles_traveled': 0.0, 'tz_shift_hours': 0,
                'altitude_flag': altitude}

    miles    = haversine_miles(prev_arena['lat'], prev_arena['lon'],
                               today_arena['lat'], today_arena['lon'])
    tz_shift = today_arena['tz_offset'] - prev_arena['tz_offset']

    penalty  = (miles / 1000.0) * 0.02
    if tz_shift > 0:
        penalty += tz_shift * 0.015
    if altitude:
        penalty += 0.04

    return {
        'fatigue_multiplier': round(max(0.88, 1.0 - penalty), 4),
        'miles_traveled':     round(miles, 1),
        'tz_shift_hours':     int(tz_shift),
        'altitude_flag':      altitude,
    }
