import pandas as pd
import numpy as np
from src.data.db import get_db_client

def check_calibration():
    db = get_db_client()
    
    with db.get_conn() as conn:
        query = """
        SELECT a.id, a.edge + c.implied_closing as predicted_prob, b.won
        FROM alerts_sent a
        JOIN bet_results b ON a.id = b.alert_id
        LEFT JOIN clv_tracking c ON a.player_name = c.player_id AND a.market = c.market
        WHERE c.implied_closing IS NOT NULL AND c.implied_closing > 0
        """
        # If we don't have clv_tracking, we must approximate predicted prob from alert implied + edge
        fallback_query = """
        SELECT a.id, 
               (1.0 / a.odds) + a.edge as predicted_prob, 
               b.won
        FROM alerts_sent a
        JOIN bet_results b ON a.id = b.alert_id
        """
        try:
            df = pd.read_sql_query(query, conn)
            if df.empty:
                df = pd.read_sql_query(fallback_query, conn)
        except:
            df = pd.read_sql_query(fallback_query, conn)
            
    if df.empty:
        print("Not enough settled data for calibration.")
        return
        
    # Drop NaNs
    df = df.dropna(subset=['predicted_prob', 'won'])
    
    if df.empty:
        print("Data exists but lacks probability fields.")
        return

    # SQLite Booleans can sometimes return as bytes objects or generic objects instead of numeric types.
    # We must explicitly cast "won" into a float for brier score arithmetic.
    def parse_won(val):
        if isinstance(val, bytes):
            return float(int.from_bytes(val, byteorder='little'))
        return float(val)
        
    df['won'] = df['won'].apply(parse_won)

    # Brier Score = 1/N * sum((predicted_prob - actual_outcome)^2)
    brier_score = np.mean((df['predicted_prob'] - df['won']) ** 2)
    
    # Bucket predictions
    bins = [0.0, 0.45, 0.50, 0.55, 0.60, 0.65, 1.0]
    labels = ['<45%', '45-50%', '50-55%', '55-60%', '60-65%', '>65%']
    df['bucket'] = pd.cut(df['predicted_prob'], bins=bins, labels=labels)
    
    calibration = df.groupby('bucket').agg(
        count=('won', 'count'),
        actual_win_rate=('won', 'mean'),
        pred_win_rate=('predicted_prob', 'mean')
    ).dropna()
    
    print("\n===========================================")
    print("      PHASE 4 PROBABILITY CALIBRATION      ")
    print("===========================================")
    print(f"Total Graded Bets: {len(df)}")
    print(f"Brier Score      : {brier_score:.4f} (0 is perfect)")
    print("-------------------------------------------")
    print("                BUCKET BREAKDOWN           ")
    print("-------------------------------------------")
    for idx, row in calibration.iterrows():
        print(f"Bucket {idx:<6} | Bets: {int(row['count']):<4} | Pred: {row['pred_win_rate']:.1%} | Actual: {row['actual_win_rate']:.1%}")
    print("===========================================\n")

if __name__ == "__main__":
    check_calibration()
