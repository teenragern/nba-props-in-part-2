import pandas as pd
from src.data.db import DatabaseClient

def analyze_market_stats():
    db = DatabaseClient()
    with db.get_conn() as conn:
        query = """
        SELECT a.market, 
               a.book, 
               (1.0 / a.odds) as implied_prob,
               c.implied_closing,
               b.won
        FROM alerts_sent a
        LEFT JOIN clv_tracking c ON a.player_name = c.player_id AND a.market = c.market
        LEFT JOIN bet_results b ON a.id = b.alert_id
        """
        try:
            df = pd.read_sql_query(query, conn)
        except:
            print("No settled market stats available.")
            return

    if df.empty:
        print("No settled market stats available.")
        return

    print("\n===========================================")
    print("      PHASE 5 MARKET MICROSTRUCTURE        ")
    print("===========================================")
    
    # 1. Bookmaker Profitability / Leader
    # Who often has the sharpest closing line correlation?
    # We measure deviation of book's initial lines from the final closing line
    df_valid = df.dropna(subset=['implied_closing']).copy()
    if not df_valid.empty:
        df_valid['book'] = df_valid['book'].astype(str).str.strip().str.lower()
        df_valid['closing_error'] = abs(df_valid['implied_prob'] - df_valid['implied_closing'])
        leaderboard = df_valid.groupby('book')['closing_error'].mean().sort_values()
        
        print("Sharp Book Leaderboard (Lowest Mean Absolute Error to Close):")
        for idx, val in leaderboard.items():
            print(f" - {idx:<15}: {val:.4f} deviation")
    else:
        print("No closing lines tracked yet for bookmaker profiling.")
        
    print("\n-------------------------------------------")
    # 2. Live Market Bias
    print("Market Systematic Bias (Model Prob - Closing Prob):")
    if not df_valid.empty:
        # We don't have model prior explicitly in this query without re-joining,
        # but we can look at actual hit rate vs market close.
        df_graded = df_valid.dropna(subset=['won']).copy()
        if not df_graded.empty:
            df_graded['bias'] = df_graded['won'].astype(float) - df_graded['implied_closing']
            bias_by_market = df_graded.groupby('market')['bias'].mean()
            for mkt, bias in bias_by_market.items():
                print(f" - {mkt}: {bias:+.2%} (Positive = Market underestimates outcome)")
        else:
            print("No graded data yet for bias check.")
    else:
        print("No closing line data yet.")
    
    print("===========================================\n")

if __name__ == "__main__":
    analyze_market_stats()
