import pandas as pd
from src.data.db import DatabaseClient
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

def generate_analytics(since: str = None):
    db = DatabaseClient()

    params = ()
    where = ""
    if since:
        where = " WHERE a.timestamp >= ?"
        params = (since,)

    with db.get_conn() as conn:
        query = f"""
        SELECT a.id, a.player_name, a.market, a.edge, b.won, b.actual_result,
               c.clv, c.closing_odds, a.odds as alert_odds
        FROM alerts_sent a
        JOIN bet_results b ON a.id = b.alert_id
        LEFT JOIN clv_tracking c ON a.player_name = c.player_id AND a.market = c.market
        {where}
        """
        df = pd.read_sql_query(query, conn, params=params)
        
    if df.empty:
        print("No settled data available for analytics.")
        return
        
    total_bets = len(df)
    wins = len(df[df['won'] == 1])
    win_rate = wins / total_bets if total_bets > 0 else 0
    
    avg_edge = df['edge'].mean()
    
    # Very crude ROI placeholder
    # Assume 1 unit bet on all
    units_won = 0.0
    for _, row in df.iterrows():
        odds = row.get('alert_odds') or 2.0
        if row['won']:
            units_won += (odds - 1.0)
        else:
            units_won -= 1.0
            
    roi = units_won / total_bets if total_bets > 0 else 0
    
    # CLV Metrics
    clv_df = df.dropna(subset=['clv'])
    avg_clv = clv_df['clv'].mean() if not clv_df.empty else 0
    clv_hit_rate = len(clv_df[clv_df['clv'] > 0]) / len(clv_df) if not clv_df.empty else 0

    print("\\n===========================================")
    print("      NBA PROP BOT MODEL PERFORMANCE       ")
    print("===========================================")
    if since:
        print(f"Filter: since {since}")
    print(f"Total Settled Alerts : {total_bets}")
    print(f"Win Rate             : {win_rate:.2%}")
    print(f"Estimated ROI        : {roi:.2%}")
    print(f"Average Advised Edge : {avg_edge:.2%}")
    print("-------------------------------------------")
    print("                CLV METRICS                ")
    print("-------------------------------------------")
    if clv_df.empty:
        print("No CLV data populated yet.")
    else:
        print(f"Average CLV Captured : {avg_clv:.2%}")
        print(f"Beat Closing Line %  : {clv_hit_rate:.2%}")
    print("===========================================\\n")
    
if __name__ == "__main__":
    generate_analytics()
