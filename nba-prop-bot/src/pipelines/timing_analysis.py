import pandas as pd
from src.data.db import get_db_client

def analyze_timing():
    db = get_db_client()
    with db.get_conn() as conn:
        query = """
        SELECT a.market, 
               c.implied_alert - c.implied_closing as edge_decay,
               (julianday(c.closing_time) - julianday(c.alert_time)) * 24 * 60 as minutes_to_tip
        FROM alerts_sent a
        JOIN clv_tracking c ON a.player_name = c.player_id AND a.market = c.market AND c.side = a.side
        WHERE c.implied_closing IS NOT NULL
        """
        try:
            df = pd.read_sql_query(query, conn)
        except Exception as e:
            print("No settled timing data available.")
            return

    if df.empty:
        print("No settled timing data available.")
        return

    print("\n===========================================")
    print("      PHASE 5 TIMING DECAY ANALYSIS        ")
    print("===========================================")
    print(f"Total Trajectories Tracked: {len(df)}")
    
    # Bucket by minutes to tip
    bins = [0, 30, 60, 120, 240, 1440]
    labels = ['<30m', '30-60m', '1-2h', '2-4h', '>4h']
    df['time_bucket'] = pd.cut(df['minutes_to_tip'], bins=bins, labels=labels)
    
    decay = df.groupby('time_bucket')['edge_decay'].mean().dropna()
    
    print("\nAverage Edge Decay by Time to Tip:")
    for idx, val in decay.items():
        print(f"{idx}: {val:.3%}")
        
    print("\nTiming Policy Suggestion:")
    for prop, group in df.groupby('market'):
        avg_decay = group['edge_decay'].mean()
        if avg_decay > 0:
            print(f"- {prop}: Decays quickly ({avg_decay:.2%}). Trigger EARLIER.")
        elif avg_decay < 0:
            print(f"- {prop}: Improves over time ({-avg_decay:.2%}). Delay alerts slightly.")
        else:
            print(f"- {prop}: Stable.")
    print("===========================================\n")

if __name__ == "__main__":
    analyze_timing()
