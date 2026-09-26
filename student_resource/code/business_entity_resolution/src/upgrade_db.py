"""Upgrade an existing DB in place: prune never-probed keys + build s1_probe.

Lets a DB built before the probe-table change be brought up to date without
re-streaming the TSVs (which is the expensive part — the raw rows and their
blocking keys are unchanged). Safe to re-run; it rebuilds from scratch.

    python src/upgrade_db.py --db work/train.db
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import blocking
from db import connect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--skip-stats", action="store_true",
                    help="key_freq already present and current.")
    ap.add_argument("--skip-prune", action="store_true")
    args = ap.parse_args()

    conn = connect(args.db)
    has_stats = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='key_freq'"
    ).fetchone()
    if not has_stats or not args.skip_stats:
        blocking.build_key_stats(conn)
    if not args.skip_prune:
        blocking.prune_unprobeable_keys(conn)
    blocking.build_probe_table(conn)
    conn.close()
    print("upgrade complete", flush=True)


if __name__ == "__main__":
    main()
