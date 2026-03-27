import argparse
from src.pipelines.sync_events import sync_events
from src.pipelines.scan_props import scan_props

def sync_all():
    print("Syncing events...")
    sync_events()
    print("Events synced.")
    
def main():
    parser = argparse.ArgumentParser(description="NBA Prop Betting Bot")
    subparsers = parser.add_subparsers(dest="command")
    
    # scan command
    scan_parser = subparsers.add_parser("scan", help="Scan pregame player props for edges")
    
    # sync command
    sync_parser = subparsers.add_parser("sync", help="Sync backend data pipelines")
    
    # update_clv command
    clv_parser = subparsers.add_parser("update_clv", help="Fetch closing lines for pending alerts")
    
    # settle command
    settle_parser = subparsers.add_parser("settle", help="Settle historical alerts using actual box scores")
    
    # stats command
    stats_parser = subparsers.add_parser("stats", help="Show performance analytics")
    
    # Phase 4 Commands
    cal_parser = subparsers.add_parser("calibration", help="Check probability calibration and Brier score")
    tune_parser = subparsers.add_parser("tune", help="Run hyperparameter tuning grid search")
    train_parser = subparsers.add_parser("train", help="Train multi-season ML models")
    
    # Phase 5 Commands
    market_stats_parser = subparsers.add_parser("market_stats", help="Show market micro-structure analytics and biases")
    steam_parser = subparsers.add_parser("steam", help="Detect active steam and line movement velocity")
    exposure_parser = subparsers.add_parser("exposure", help="Check current portfolio exposure against bankroll limits")
    timing_parser = subparsers.add_parser("timing_analysis", help="Analyze average edge decay timing to tip-off")
    
    # Continuous Scheduler
    run_parser = subparsers.add_parser("run", help="Start the continuous automated scheduler")
    
    args = parser.parse_args()
    
    if args.command == "run":
        from src.pipelines.run_scheduler import start_scheduler
        start_scheduler()
    elif args.command == "scan":
        scan_props()
    elif args.command == "sync":
        sync_all()
    elif args.command == "update_clv":
        from src.pipelines.update_clv import update_clv_lines
        update_clv_lines()
    elif args.command == "settle":
        from src.pipelines.settle_results import settle_alerts
        settle_alerts()
    elif args.command == "stats":
        from src.pipelines.analytics import generate_analytics
        generate_analytics()
    elif args.command == "calibration":
        from src.pipelines.calibration import check_calibration
        check_calibration()
    elif args.command == "tune":
        from src.pipelines.tune import run_tuning
        from src.data.db import DatabaseClient
        run_tuning(DatabaseClient())
    elif args.command == "train":
        from src.pipelines.train_ml import train_ml_models
        train_ml_models()
    elif args.command == "market_stats":
        from src.pipelines.market_stats import analyze_market_stats
        analyze_market_stats()
    elif args.command == "steam":
        from src.pipelines.steam import check_steam
        check_steam()
    elif args.command == "exposure":
        from src.pipelines.exposure import check_exposure
        check_exposure()
    elif args.command == "timing_analysis":
        from src.pipelines.timing_analysis import analyze_timing
        analyze_timing()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
