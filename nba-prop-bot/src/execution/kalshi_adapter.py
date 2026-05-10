import os
from typing import Dict, Any, Optional

from src.clients.exchange_client import ExchangeClient
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Map standard markets to Kalshi series prefixes and title suffixes
_MARKET_MAP = {
    'player_points':   ('KXNBAPTS', 'points'),
    'player_rebounds': ('KXNBAREB', 'rebounds'),
    'player_assists':  ('KXNBAAST', 'assists'),
    'player_threes':   ('KXNBA3PT', '3-pointers'),
}


def _match_prop_ticker(
    client: ExchangeClient,
    player_name: str,
    market: str,
    line: float
) -> Optional[str]:
    """
    Find the Kalshi ticker for a specific player prop and line.
    Kalshi lines are typically integers representing "X+ stats" (which means OVER X-0.5).
    For example, a line of 25.5 points corresponds to "26+ points" on Kalshi.
    """
    if market not in _MARKET_MAP:
        return None
        
    series, suffix = _MARKET_MAP[market]
    
    # Standard decimal lines (e.g. 25.5) map to the next integer (26)
    target_int = int(line + 0.5)
    
    # Kalshi titles look like: "Shai Gilgeous-Alexander: 26+ points"
    player_last_name = player_name.split()[-1].lower()
    
    # To avoid fetching all markets every time, we can search specifically
    markets = client.search_markets(player_last_name, limit=50)
    
    for m in markets:
        ticker = m.get('ticker', '')
        title = m.get('title', '').lower()
        
        # Check if it matches our market series and target stat
        if series in ticker and f"{target_int}+ {suffix}" in title and player_last_name in title:
            return ticker
            
    return None


def try_place_kalshi_trade(
    edge_data: Dict[str, Any],
    max_stake: float
) -> Optional[Dict[str, Any]]:
    """
    Attempt to place a trade on Kalshi for the given edge.
    Returns a dict with execution details if successful, None if it failed or was skipped.
    """
    client = ExchangeClient()
    if not client.enabled:
        logger.warning("Kalshi client is not enabled. Cannot place live trade.")
        return None

    player = edge_data.get('player_id') or edge_data.get('player_name', '')
    market = edge_data.get('market', '')
    side = edge_data.get('side', '').upper()
    line = float(edge_data.get('line', 0.0))
    model_prob = float(edge_data.get('model_prob', 0.0))
    
    # Only support specific integer+0.5 lines since Kalshi uses X+ contracts
    if line % 1 != 0.5:
        logger.debug(f"Kalshi adapter: skipping non-half-point line {line}")
        return None

    ticker = _match_prop_ticker(client, player, market, line)
    if not ticker:
        logger.debug(f"Kalshi adapter: no matching ticker found for {player} {market} {line}")
        return None

    market_price = client.get_market_price(ticker)
    if not market_price:
        logger.debug(f"Kalshi adapter: could not fetch prices for {ticker}")
        return None

    min_edge = float(os.getenv('EXCHANGE_ARB_MIN_EDGE', '0.03'))
    
    # Determine which contract to buy (YES for OVER, NO for UNDER)
    if side == 'OVER':
        contract_side = 'yes'
        true_prob = model_prob
        ask_price = market_price.get('yes_ask', 0.0)
    elif side == 'UNDER':
        contract_side = 'no'
        true_prob = 1.0 - model_prob
        ask_price = market_price.get('no_ask', 0.0)
    else:
        return None

    if ask_price <= 0.0 or ask_price >= 1.0:
        logger.debug(f"Kalshi adapter: invalid ask price {ask_price} for {ticker}")
        return None

    edge = true_prob - ask_price
    if edge < min_edge:
        logger.info(f"Kalshi adapter: edge {edge:.3f} is below min {min_edge} for {ticker}")
        return None

    # We have a profitable edge! Determine limit price and place order.
    # Place limit order at the current ask to get an immediate fill.
    limit_cents = int(ask_price * 100)
    
    logger.info(f"Kalshi adapter: placing {contract_side.upper()} order on {ticker} at {limit_cents}c (edge: {edge:.3f})")
    
    order_response = client.place_order(
        ticker=ticker,
        side=contract_side,
        limit_price_cents=limit_cents,
        max_stake_dollars=max_stake
    )
    
    if order_response and order_response.get('order'):
        order = order_response['order']
        # Convert fill price back to decimal odds for internal tracking
        fill_odds = 1.0 / (limit_cents / 100.0)
        return {
            'order_id': order.get('order_id'),
            'fill_odds': fill_odds,
            'book': 'kalshi',
            'status': order.get('status', 'resting')
        }
    
    logger.warning(f"Kalshi adapter: order placement failed for {ticker}")
    return None
