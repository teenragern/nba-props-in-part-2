"""
SQLite backup pipeline.

Performs a hot online backup using sqlite3.Connection.backup() (Python ≥ 3.7),
which is safe under concurrent writes — no read lock on the live DB.

Config (env vars):
  DB_PATH            — source database (default: props.db)
  BACKUP_DIR         — local directory for backup files (default: backups/)
  BACKUP_KEEP_DAYS   — how many daily backups to retain (default: 7)
  BACKUP_S3_BUCKET   — if set, also upload to s3://<bucket>/backups/
  BACKUP_STATS_DB    — if set, also backs up the stats_cache.db at this path
"""

import os
import sqlite3
from datetime import datetime, timedelta, timezone

from src.config import DB_PATH
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

BACKUP_DIR       = os.getenv('BACKUP_DIR', 'backups')
BACKUP_KEEP_DAYS = int(os.getenv('BACKUP_KEEP_DAYS', '7'))
S3_BUCKET        = os.getenv('BACKUP_S3_BUCKET', '')
STATS_DB_PATH    = os.getenv('BACKUP_STATS_DB', '')


def _hot_backup(src_path: str, dest_path: str) -> float:
    """Copy src SQLite DB to dest using the safe online backup API. Returns size in MB."""
    src  = sqlite3.connect(src_path)
    dest = sqlite3.connect(dest_path)
    try:
        src.backup(dest)
    finally:
        dest.close()
        src.close()
    return os.path.getsize(dest_path) / (1024 * 1024)


def _s3_upload(local_path: str, s3_key: str) -> bool:
    try:
        import boto3  # type: ignore
        boto3.client('s3').upload_file(local_path, S3_BUCKET, s3_key)
        logger.info(f"Uploaded to s3://{S3_BUCKET}/{s3_key}")
        return True
    except Exception as e:
        logger.warning(f"S3 upload failed ({s3_key}): {e}")
        return False


def _purge_old_backups(prefix: str):
    cutoff = datetime.now(timezone.utc) - timedelta(days=BACKUP_KEEP_DAYS)
    for fname in os.listdir(BACKUP_DIR):
        if not fname.startswith(prefix) or not fname.endswith('.db'):
            continue
        fpath = os.path.join(BACKUP_DIR, fname)
        mtime = datetime.fromtimestamp(os.path.getmtime(fpath), tz=timezone.utc)
        if mtime < cutoff:
            try:
                os.remove(fpath)
                logger.info(f"Purged old backup: {fname}")
            except Exception as e:
                logger.warning(f"Could not purge {fname}: {e}")


def backup_db() -> dict:
    """
    Back up props.db (and optionally stats_cache.db) to BACKUP_DIR.
    Returns a summary dict with paths and sizes.
    """
    os.makedirs(BACKUP_DIR, exist_ok=True)
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    results = {}

    # ── props.db ────────────────────────────────────────────────────────────
    props_dest = os.path.join(BACKUP_DIR, f'props.{today}.db')
    try:
        size_mb = _hot_backup(DB_PATH, props_dest)
        logger.info(f"DB backup written: {props_dest} ({size_mb:.1f} MB)")
        results['props'] = {'path': props_dest, 'size_mb': round(size_mb, 2)}
        if S3_BUCKET:
            _s3_upload(props_dest, f'backups/props.{today}.db')
    except Exception as e:
        logger.error(f"props.db backup failed: {e}")
        results['props'] = {'error': str(e)}

    # ── stats_cache.db (optional) ────────────────────────────────────────────
    if STATS_DB_PATH and os.path.exists(STATS_DB_PATH):
        stats_dest = os.path.join(BACKUP_DIR, f'stats_cache.{today}.db')
        try:
            size_mb = _hot_backup(STATS_DB_PATH, stats_dest)
            logger.info(f"Stats DB backup written: {stats_dest} ({size_mb:.1f} MB)")
            results['stats'] = {'path': stats_dest, 'size_mb': round(size_mb, 2)}
            if S3_BUCKET:
                _s3_upload(stats_dest, f'backups/stats_cache.{today}.db')
        except Exception as e:
            logger.warning(f"stats_cache.db backup failed: {e}")
            results['stats'] = {'error': str(e)}

    # ── Retention cleanup ────────────────────────────────────────────────────
    _purge_old_backups('props.')
    if STATS_DB_PATH:
        _purge_old_backups('stats_cache.')

    return results
