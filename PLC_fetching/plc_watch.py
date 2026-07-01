"""
plc_watch.py — find which PLC values actually change.

Connects once, discovers every Data Block present on the PLC, then watches them
for a fixed window and reports which 16-bit words moved. Use it to tell static
config/setpoint data apart from live process signals, and to spot DBs that the
logger (plc_comms.py) is not capturing yet.

Note: this only sees Data Blocks. If nothing here moves, the live signals are
probably in Merker (M), Input (I) or Output (Q) memory, which needs read_area
rather than db_read — say so and that can be added.

    python plc_watch.py                       # discover 1..255, watch 30 s
    python plc_watch.py --duration 60 --interval 0.5
"""

import argparse
import logging
import time

from snap7.client import Client
from snap7.util import get_int

DEFAULT_IP = "200.200.5.1"
DEFAULT_RACK = 0
DEFAULT_SLOT = 2
INT_BYTES = 2

# DBs the logger currently captures, so we can flag anything we're missing.
LOGGED_DBS = {21, 22, 23, 30, 31, 40, 41, 42, 43, 44, 45, 46, 47, 48}


def db_size(client: Client, db: int, hi: int = 512) -> int:
    """Largest readable length (bytes) at offset 0 = the DB's size. 0 if absent."""
    last_ok, size = 0, 2
    while size <= hi:                       # grow until a read fails
        try:
            client.db_read(db, 0, size)
            last_ok, size = size, size * 2
        except Exception:
            break
    lo, high = last_ok, min(size, hi)
    while lo + 1 < high:                    # binary-search the exact edge
        mid = (lo + high) // 2
        try:
            client.db_read(db, 0, mid)
            lo = mid
        except Exception:
            high = mid
    return lo


def discover(client: Client, lo: int, hi: int) -> dict[int, int]:
    """Return {db_number: size_bytes} for every DB that exists in [lo, hi]."""
    logging.disable(logging.ERROR)          # hush per-probe error spam
    try:
        present = {}
        for db in range(lo, hi + 1):
            try:
                client.db_read(db, 0, INT_BYTES)
            except Exception:
                continue
            present[db] = db_size(client, db)
        return present
    finally:
        logging.disable(logging.NOTSET)


def watch(client: Client, sizes: dict[int, int], duration: float, interval: float):
    """Poll every DB for `duration` seconds, tracking per-word change stats."""
    stats: dict[tuple[int, int], dict] = {}
    deadline = time.monotonic() + duration
    polls = 0
    while time.monotonic() < deadline:
        for db, size in sizes.items():
            try:
                raw = client.db_read(db, 0, size)
            except Exception as e:
                logging.warning(f"DB{db} read failed: {e}")
                continue
            for word in range(size // INT_BYTES):
                value = get_int(raw, word * INT_BYTES)
                key = (db, word)
                stat = stats.get(key)
                if stat is None:
                    stats[key] = {"first": value, "last": value,
                                  "min": value, "max": value, "changed": False}
                else:
                    if value != stat["last"]:
                        stat["changed"] = True
                    stat["last"] = value
                    stat["min"] = min(stat["min"], value)
                    stat["max"] = max(stat["max"], value)
        polls += 1
        time.sleep(interval)
    return stats, polls


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Find which PLC DB words change over time.")
    parser.add_argument("--ip", default=DEFAULT_IP)
    parser.add_argument("--rack", type=int, default=DEFAULT_RACK)
    parser.add_argument("--slot", type=int, default=DEFAULT_SLOT)
    parser.add_argument("--lo", type=int, default=1, help="lowest DB number to probe")
    parser.add_argument("--hi", type=int, default=255, help="highest DB number to probe")
    parser.add_argument("--duration", type=float, default=30.0, help="watch window in seconds")
    parser.add_argument("--interval", type=float, default=0.5, help="poll interval in seconds")
    args = parser.parse_args()

    client = Client()
    client.connect(args.ip, args.rack, args.slot)
    try:
        sizes = discover(client, args.lo, args.hi)
        print(f"\nPresent DBs ({len(sizes)}): "
              + ", ".join(f"DB{db}({sz}B)" for db, sz in sorted(sizes.items())))
        missing = sorted(set(sizes) - LOGGED_DBS)
        if missing:
            print(f"NOT in plc_comms logger yet: {', '.join(f'DB{d}' for d in missing)}")

        print(f"\nWatching for {args.duration:g}s at {args.interval:g}s intervals…")
        stats, polls = watch(client, sizes, args.duration, args.interval)

        changed = sorted(k for k, s in stats.items() if s["changed"])
        print(f"\n{polls} polls. {len(changed)} of {len(stats)} words changed.\n")
        if not changed:
            print("Nothing moved. These DBs are static (config/setpoint) or the line is "
                  "idle. Live signals are likely in other DBs or in M/I/Q memory "
                  "(needs read_area, not db_read).")
        else:
            print(f"{'DB.word':>10} {'first':>8} {'min':>8} {'max':>8} {'last':>8}")
            for db, word in changed:
                s = stats[(db, word)]
                print(f"{f'DB{db}.{word}':>10} {s['first']:>8} {s['min']:>8} "
                      f"{s['max']:>8} {s['last']:>8}")
    finally:
        if client.get_connected():
            client.disconnect()


if __name__ == "__main__":
    main()
