import pandas as pd
from src.data.db import DatabaseClient
from src.config import BANKROLL

def check_exposure():
    db = DatabaseClient()
    MAX_DAILY_RISK = BANKROLL * 0.50
    MAX_PER_GAME = BANKROLL * 0.10
    
    with db.get_conn() as conn:
        query = """
        SELECT player_name, market, side, odds, stake, timestamp
        FROM alerts_sent
        WHERE date(timestamp) = date('now')
        """
        try:
            df = pd.read_sql_query(query, conn)
        except:
            print("No alerts found for today.")
            return

    print("\n===========================================")
    print("      PHASE 5 PORTFOLIO EXPOSURE           ")
    print("===========================================")
    
    if df.empty:
        print(f"Total Daily Risk: $0.00 / ${MAX_DAILY_RISK:.2f}")
        print("No exposure today.")
        print("===========================================\n")
        return
        
    total_risk = df['stake'].sum()
    print(f"Total Daily Risk: ${total_risk:.2f} / ${MAX_DAILY_RISK:.2f} ({(total_risk/MAX_DAILY_RISK):.1%} of limit)")
    print(f"Current Bankroll: ${BANKROLL:.2f}")
    
    print("\n-------------------------------------------")
    print("                OPEN POSITIONS             ")
    print("-------------------------------------------")
    for _, row in df.iterrows():
        print(f" - {row['player_name']} {row['market']} {row['side']} @ {row['odds']} | Stake: ${row['stake']:.2f}")
    print("===========================================\n")

if __name__ == "__main__":
    check_exposure()
