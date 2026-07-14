"""
plc_comms.py — stream named live signals from a Siemens S7-300 to a rolling CSV.

Protocol
--------
Uses `python-snap7` over Ethernet (ISO-on-TCP, RFC 1006, TCP port 102). The
S7-300 CPU is at rack 0, slot 2 (slot 1 = power supply). One connection is
opened and reused.

What it reads
-------------
The signals to log come from the machine's exported symbol table
(L14_shrinkwrapper_symbol_table_EN.csv: columns Symbol, Address, DataType, ...).
Each symbol's Address says where the value lives and DataType says how to decode
it. By DEFAULT every named signal is logged — including the digital bits (M/I/Q),
which is where the fault/state signals live (fallen product, jams, axis faults):
    counters   C n           -> CT area, BCD-decoded
    mem words  MW n / MD n   -> MK area, INT/WORD/DINT/DWORD/REAL
    analog     PIW n / PQW n -> PE / PA area, INT/WORD
    bits       M/I/Q .b      -> MK / PE / PA area, BOOL (faults, states)
    timers     T n           -> TM area
    DB words   DB99.DBW154   -> DB area, INT/WORD/DINT/REAL/BOOL
Use --lite for the numeric-only subset (counters/analog/words, no bits/timers).
Whole data blocks are still skipped, but a symbol-table row that names a
specific DB word/bit (e.g. Address "DB99.DBW154", DataType INT) is logged like
any other numeric signal — that's where recipe/format setpoints live.

Each CSV column header is the PLC address plus an English description (or the
symbol as shorthand when there's no comment), e.g. "M115.7_Product_lying_down" or
"C7_number_of_packs". So a column maps straight back to the address a colleague
would quote (M115.7, M100.7, ...).

Reads are bulk: one read_area per memory area per poll (not one per signal),
then each signal is sliced out of the buffer. At startup each area is probed
once; an area the PLC or library rejects is dropped with a warning, and the rest
keep logging.

Storage (change-only + rolling MB cap)
--------------------------------------
A row is written only when at least one value differs from the last stored row
(change-of-value). Idle line -> the file barely grows. Set --heartbeat N to also
write an unchanged row every N seconds; --store-all to record every poll. Rows
append to numbered CSV segments; when all segments exceed --max-mb the oldest are
deleted, holding roughly the most recent --max-mb. Each row holds until the next:
forward-fill on load (pandas: df.set_index("timestamp").ffill()).

Usage
-----
    python plc_comms.py                              # logs EVERYTHING (default)
    python plc_comms.py --heartbeat 300              # everything + 5-min liveness row
    python plc_comms.py --lite                       # numeric-only (no bits/timers)
    python plc_comms.py --symbols path/to/table.csv  # custom symbol table
Stop with Ctrl+C.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import struct
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from snap7.client import Client

# --- PLC endpoint ---------------------------------------------------------
DEFAULT_IP = "200.200.5.1"
DEFAULT_RACK = 0
DEFAULT_SLOT = 2  # S7-300 CPU is in slot 2 (slot 1 = power supply)
_HERE = Path(__file__).resolve().parent
DEFAULT_SYMBOLS = str(_HERE / "L14_shrinkwrapper_symbol_table_EN.csv")
DEFAULT_DATA_DIR = _HERE / "output"   # captured CSV logs land here regardless of cwd

# --- S7 memory area codes -------------------------------------------------
# Prefer the library's Area enum; fall back to raw S7 codes if absent.
try:  # python-snap7 2.x
    from snap7.type import Area as _Area
except Exception:  # noqa: BLE001 - older layout / fork
    try:
        from snap7.types import Area as _Area
    except Exception:  # noqa: BLE001
        _Area = None

if _Area is not None:
    AREA = {"PE": _Area.PE, "PA": _Area.PA, "MK": _Area.MK,
            "CT": _Area.CT, "TM": _Area.TM, "DB": _Area.DB}
else:
    AREA = {"PE": 0x81, "PA": 0x82, "MK": 0x83, "CT": 0x1C, "TM": 0x1D, "DB": 0x84}

COUNTED_AREAS = {"CT", "TM"}  # read_area size is in elements (x2 bytes), not bytes


# --- Big-endian decoders (S7 is big-endian; self-contained, no snap7.util) ---
def _u16(b, o): return int.from_bytes(b[o:o + 2], "big", signed=False)
def _i16(b, o): return int.from_bytes(b[o:o + 2], "big", signed=True)
def _u32(b, o): return int.from_bytes(b[o:o + 4], "big", signed=False)
def _i32(b, o): return int.from_bytes(b[o:o + 4], "big", signed=True)
def _f32(b, o): return struct.unpack_from(">f", b, o)[0]
def _bool(b, o, bit): return bool((b[o] >> bit) & 1)


def _counter(b, o):
    """S7 counter value: 3-digit BCD in the low 12 bits of the word."""
    w = _u16(b, o)
    return ((w >> 8) & 0xF) * 100 + ((w >> 4) & 0xF) * 10 + (w & 0xF)


# DataType -> (decode key, width in bytes)
_DTYPE = {
    "WORD": ("word", 2), "INT": ("int", 2),
    "DWORD": ("dword", 4), "DINT": ("dint", 4), "REAL": ("real", 4),
    "BOOL": ("bool", 1), "COUNTER": ("counter", 2), "TIMER": ("timer", 2),
}
# Address prefix -> area code
_PREFIX_AREA = {
    "C": "CT", "T": "TM",
    "M": "MK", "MW": "MK", "MD": "MK", "MB": "MK",
    "I": "PE", "E": "PE", "PIW": "PE", "PIB": "PE", "PID": "PE", "EW": "PE",
    "Q": "PA", "A": "PA", "PQW": "PA", "PQB": "PA", "PQD": "PA", "AW": "PA",
}

# Data-block word/bit address, e.g. "DB99.DBW154", "DB 99.DBX 10.3" (spaces
# stripped first). Groups: db number, sub-type (DBW/DBD/DBB/DBX), byte offset,
# optional bit for DBX. Whole-block rows like "DB 99" (no DBx part) don't match
# and stay skipped, as before.
_DB_ADDR_RE = re.compile(r"^DB(\d+)\.?(DB[WXBD])(\d+)(?:\.(\d+))?$")


@dataclass(frozen=True)
class Signal:
    name: str          # column header (the PLC symbol)
    area: str          # 'CT' | 'TM' | 'MK' | 'PE' | 'PA' | 'DB'
    start: int         # byte offset, or element index for CT/TM
    bit: Optional[int] # bit number for BOOL, else None
    width: int         # bytes
    decode: str        # 'int'|'word'|'dint'|'dword'|'real'|'bool'|'counter'|'timer'
    db: Optional[int] = None  # data-block number for area=='DB', else None


# --- Symbol table -> signals ---------------------------------------------
def normalize_watch(tokens) -> set[str]:
    """Normalize watch entries (addresses or symbol names) for matching."""
    return {t.upper().replace(" ", "") for t in tokens if t.strip()}


def _clean_desc(text: str, maxlen: int = 45) -> str:
    """ASCII-safe, underscore-joined description fragment for a column header.

    Non-ASCII-alphanumeric chars (spaces, punctuation, lost accents shown as the
    replacement char) become underscores, which are then collapsed.
    """
    cleaned = "".join(ch if (ch.isascii() and ch.isalnum()) else "_" for ch in (text or ""))
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")[:maxlen].strip("_")


def load_name_overrides(path: str | Path) -> dict[str, str]:
    """Read a CSV with Address + Header columns into {NORMALIZED_ADDRESS: header}."""
    overrides: dict[str, str] = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            addr = (row.get("Address") or "").strip().upper().replace(" ", "")
            header = (row.get("Header") or "").replace(",", "_").replace("\n", " ").strip()
            if addr and header:
                overrides[addr] = header
    return overrides


def _column_name(addr_token: str, entry: dict, symbol: str,
                 name_overrides, seen: set[str], n_signals: int) -> str:
    """Build a unique CSV column header for one signal.

    Header = PLC address (what colleagues quote, e.g. M115.7 or DB99.DBW154) +
    an English description, or the symbol as shorthand when there's no comment.
    So a column always maps straight back to the address. A --names CSV can
    override it, and a numeric suffix keeps headers unique.
    """
    desc = _clean_desc(entry.get("Comment_EN")) or _clean_desc(symbol)
    name = f"{addr_token}_{desc}" if desc else addr_token
    if name_overrides:  # manual rename via --names CSV wins
        name = name_overrides.get(addr_token.upper(), name)
    if name in seen:  # keep column headers unique
        name = f"{name}_{n_signals}"
    seen.add(name)
    return name


def parse_symbols(path: str | Path, include_bits: bool = False,
                  include_timers: bool = False, watch=None,
                  name_overrides=None) -> list[Signal]:
    """
    Build the signal list from the exported symbol table CSV.

    `watch` is a set of normalized addresses ("M115.7") or symbol names that are
    always included even when their type would otherwise be opt-in (bits/timers).

    Data-block words are supported when the address names a specific word/bit,
    e.g. "DB99.DBW154" with DataType INT. Whole-block rows ("DB 99") stay skipped.
    """
    watch = watch or set()
    signals: list[Signal] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8-sig", newline="") as f:
        for entry in csv.DictReader(f):
            raw_addr = (entry.get("Address") or "").strip()
            dtype = (entry.get("DataType") or "").strip().upper()
            symbol = (entry.get("Symbol") or "").strip()
            if not raw_addr or dtype not in _DTYPE or not symbol:
                continue

            # --- Data-block word/bit tag, e.g. "DB99.DBW154" -----------------
            db_match = _DB_ADDR_RE.match(raw_addr.upper().replace(" ", ""))
            if db_match:
                db_num, sub, off_s, bit_s = db_match.groups()
                decode, width = _DTYPE[dtype]
                start = int(off_s)
                bit = int(bit_s) if bit_s is not None else None
                addr_token = f"DB{db_num}.{sub}{start}" + (f".{bit}" if bit is not None else "")
                forced = addr_token.upper() in watch or symbol.upper() in watch
                if decode == "bool" and not include_bits and not forced:
                    continue  # DB bit under --lite, unless watched
                name = _column_name(addr_token, entry, symbol, name_overrides, seen, len(signals))
                signals.append(Signal(name, "DB", start, bit, width, decode, db=int(db_num)))
                continue

            # --- Merker / counter / timer / analog / IO ----------------------
            addr = raw_addr.split()
            if len(addr) < 2:
                continue
            area = _PREFIX_AREA.get(addr[0].upper())
            if area is None:
                continue  # not a memory area we read (whole DB/FB/FC/OB/...)

            decode, width = _DTYPE[dtype]
            # A watched address/name overrides the bit/timer opt-in gate.
            forced = (addr[0] + addr[1]).upper() in watch or symbol.upper() in watch
            if decode == "bool" and not include_bits and not forced:
                continue
            if decode == "timer" and not include_timers and not forced:
                continue

            num = addr[1]
            if "." in num:  # bit address, e.g. M 10.0
                byte_s, bit_s = num.split(".", 1)
                start, bit = int(byte_s), int(bit_s)
            else:
                start, bit = int(num), None

            addr_token = addr[0] + num
            name = _column_name(addr_token, entry, symbol, name_overrides, seen, len(signals))
            signals.append(Signal(name, area, start, bit, width, decode))
    return signals


def plan_reads(signals: list[Signal]) -> dict[str, dict]:
    """Group signals into one bulk read span each.

    Grouped by memory area, except data blocks: each DB is its own address
    space, so it gets its own group keyed "DB<n>" (e.g. "DB99"). Each group
    records the snap7 area code and db number needed for the read.
    """
    plan: dict[str, dict] = {}
    for sig in signals:
        key = f"DB{sig.db}" if sig.area == "DB" else sig.area
        grp = plan.setdefault(
            key, {"signals": [], "base": None, "size": 0,
                  "area": sig.area, "db": sig.db or 0})
        grp["signals"].append(sig)
    for key, grp in plan.items():
        area = grp["area"]
        starts = [s.start for s in grp["signals"]]
        base = min(starts)
        grp["base"] = base
        if area in COUNTED_AREAS:
            grp["size"] = max(starts) - base + 1            # element count
        else:
            grp["size"] = max(s.start + s.width for s in grp["signals"]) - base
    return plan


def _decode(sig: Signal, buf: bytes, base: int, area: str):
    """Pull one signal's value out of its area buffer."""
    off = (sig.start - base) * 2 if area in COUNTED_AREAS else sig.start - base
    if sig.decode == "bool":
        return _bool(buf, off, sig.bit)
    if sig.decode == "int":
        return _i16(buf, off)
    if sig.decode == "word":
        return _u16(buf, off)
    if sig.decode == "dint":
        return _i32(buf, off)
    if sig.decode == "dword":
        return _u32(buf, off)
    if sig.decode == "real":
        return _f32(buf, off)
    if sig.decode == "counter":
        return _counter(buf, off)
    return _u16(buf, off)  # timer: raw word


_MAX_READ_BYTES = 200  # stay under the ~222-byte usable payload of a 240-byte PDU


def _read_area(client: Client, area: str, base: int, size: int, db: int = 0) -> bytes:
    """
    Read a memory area in PDU-safe chunks and concatenate.

    A single S7 read can't exceed the negotiated PDU (~222 usable bytes). The C
    snap7 library auto-splits larger reads; a pure-Python one may not, so we
    split here and stay correct either way. For CT/TM, `size` is an element
    count (each element is 2 bytes), so the step is halved. `db` is the data
    block number when area == 'DB' (ignored otherwise).
    """
    counted = area in COUNTED_AREAS
    step = (_MAX_READ_BYTES // 2) if counted else _MAX_READ_BYTES
    out = bytearray()
    offset = 0
    while offset < size:
        n = min(step, size - offset)
        out += client.read_area(AREA[area], db, base + offset, n)
        offset += n
    return bytes(out)


def column_names(signals: list[Signal]) -> list[str]:
    return ["timestamp"] + [s.name for s in signals]


def read_signals(client: Client, plan: dict[str, dict]) -> tuple[dict, int]:
    """One bulk read per area; decode every signal. Returns (row, ok_area_count)."""
    row: dict = {"timestamp": datetime.now().isoformat(timespec="milliseconds")}
    ok_areas = 0
    for key, grp in plan.items():
        area = grp["area"]
        try:
            buf = _read_area(client, area, grp["base"], grp["size"], grp["db"])
            ok_areas += 1
        except Exception as e:
            logging.warning(f"read_area {key} failed: {e}")
            buf = None
        for sig in grp["signals"]:
            if buf is None:
                row[sig.name] = None
                continue
            try:
                row[sig.name] = _decode(sig, buf, grp["base"], area)
            except Exception as e:
                logging.warning(f"decode {sig.name} ({key}) failed: {e}")
                row[sig.name] = None
    return row, ok_areas


def probe_areas(client: Client, plan: dict[str, dict]) -> dict[str, dict]:
    """Drop any area that can't be read, with a clear warning. Raise if none work."""
    usable = {}
    for key, grp in plan.items():
        try:
            _read_area(client, grp["area"], grp["base"], grp["size"], grp["db"])
            usable[key] = grp
            logging.info(f"area {key}: {len(grp['signals'])} signals OK")
        except Exception as e:
            logging.warning(
                f"area {key} not readable ({e}); skipping {len(grp['signals'])} signals"
            )
    if not usable:
        raise RuntimeError("No memory area is readable on this PLC/library")
    return usable


# --- Rolling CSV store ----------------------------------------------------
class RollingCsvStore:
    """Append rows to CSV segment files, keeping total on-disk size under a cap."""

    SEGMENT_GLOB = "plc_*.csv"

    def __init__(self, directory, columns, max_mb=100.0, segment_mb=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.columns = list(columns)
        self._header = ",".join(self.columns) + "\n"
        self._header_bytes = len(self._header.encode("utf-8"))

        self.max_bytes = int(max_mb * 1024 * 1024)
        if segment_mb is None:
            segment_mb = max(max_mb / 10.0, 0.25)
        self.segment_bytes = int(segment_mb * 1024 * 1024)
        if self.segment_bytes >= self.max_bytes:  # need >=2 segments to evict
            self.segment_bytes = max(self.max_bytes // 2, 1)

        self._seq = self._highest_existing_seq()
        self._fh = None
        self._bytes = 0
        self._current_path: Optional[Path] = None
        self._open_new_segment()
        self._enforce_budget()

    def _segment_files(self) -> list[Path]:
        return sorted(self.directory.glob(self.SEGMENT_GLOB))

    def _highest_existing_seq(self) -> int:
        seqs = []
        for path in self.directory.glob(self.SEGMENT_GLOB):
            try:
                seqs.append(int(path.stem.split("_")[1]))
            except (IndexError, ValueError):
                continue
        return max(seqs, default=0)

    def _open_new_segment(self) -> None:
        if self._fh is not None:
            self._fh.close()
        self._seq += 1
        self._current_path = self.directory / f"plc_{self._seq:08d}.csv"
        self._fh = open(self._current_path, "w", newline="", encoding="utf-8")
        self._fh.write(self._header)
        self._fh.flush()
        self._bytes = self._header_bytes

    def append(self, row: dict) -> None:
        line = ",".join(self._format(row.get(c)) for c in self.columns) + "\n"
        encoded_len = len(line.encode("utf-8"))
        if (self._bytes > self._header_bytes
                and self._bytes + encoded_len > self.segment_bytes):
            self._open_new_segment()
        self._fh.write(line)
        self._fh.flush()
        self._bytes += encoded_len
        # Enforce the cap every append, not only at rotation.
        self._enforce_budget()

    @staticmethod
    def _format(value) -> str:
        return "" if value is None else str(value)

    def _enforce_budget(self) -> None:
        segments = self._segment_files()
        total = sum(p.stat().st_size for p in segments)
        for path in segments:
            if total <= self.max_bytes:
                break
            if path == self._current_path:
                continue
            try:
                size = path.stat().st_size
                path.unlink()
                total -= size
                logging.info(f"Rolling store: evicted {path.name} ({size / 1e6:.1f} MB)")
            except OSError as e:
                logging.warning(f"Could not evict {path.name}: {e}")

    def close(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None


# --- Connection helpers ---------------------------------------------------
def _try_connect(client: Client, ip: str, rack: int, slot: int) -> bool:
    try:
        if not client.get_connected():
            client.connect(ip, rack, slot)
        return True
    except Exception as e:
        logging.warning(f"Connect to {ip} (rack {rack}, slot {slot}) failed: {e}")
        return False


def _reconnect(client: Client, ip: str, rack: int, slot: int) -> bool:
    try:
        if client.get_connected():
            client.disconnect()
    except Exception as e:
        logging.debug(f"Disconnect during reconnect ignored: {e}")
    return _try_connect(client, ip, rack, slot)


# --- Stream loop ----------------------------------------------------------
def stream(
    signals: list[Signal],
    *,
    ip: str = DEFAULT_IP,
    rack: int = DEFAULT_RACK,
    slot: int = DEFAULT_SLOT,
    directory: str | Path = DEFAULT_DATA_DIR,
    interval: float = 1.0,
    max_mb: float = 100.0,
    segment_mb: Optional[float] = None,
    store_on_change: bool = True,
    heartbeat_s: float = 0.0,
    max_polls: Optional[int] = None,
    max_fail_streak: int = 60,
) -> int:
    """Poll the named signals and append change-of-value rows to a rolling CSV."""
    if not signals:
        raise ValueError("no signals to log")

    plan = plan_reads(signals)
    store = RollingCsvStore(directory, column_names(signals), max_mb=max_mb, segment_mb=segment_mb)
    client = Client()
    polls = stored = 0
    last_values: Optional[tuple] = None
    last_store_t: Optional[float] = None
    try:
        connected = _try_connect(client, ip, rack, slot)
        for _ in range(4):
            if connected:
                break
            time.sleep(2.0)
            connected = _try_connect(client, ip, rack, slot)
        if not connected:
            raise ConnectionError(f"Cannot reach PLC at {ip}:102 (rack {rack}, slot {slot})")

        plan = probe_areas(client, plan)
        active = [s for grp in plan.values() for s in grp["signals"]]
        value_columns = [s.name for s in active]
        logging.info(f"Connected to {ip}; logging {len(active)} signals to {store.directory}")

        fail_streak = 0
        while max_polls is None or polls < max_polls:
            cycle_start = time.monotonic()

            row, ok_areas = read_signals(client, plan)
            polls += 1
            if ok_areas == 0:
                fail_streak += 1
                logging.error(f"All area reads failed (streak {fail_streak}); reconnecting…")
                _reconnect(client, ip, rack, slot)
                if fail_streak >= max_fail_streak:
                    raise ConnectionError(
                        f"No reads succeeded in {fail_streak} consecutive cycles; giving up"
                    )
            else:
                fail_streak = 0
                values = tuple(row[c] for c in value_columns)
                changed = last_values is None or values != last_values
                heartbeat_due = (
                    heartbeat_s > 0
                    and last_store_t is not None
                    and cycle_start - last_store_t >= heartbeat_s
                )
                if not store_on_change or changed or heartbeat_due:
                    store.append(row)
                    stored += 1
                    last_values = values
                    last_store_t = cycle_start
                    if stored % 100 == 0:
                        logging.info(f"{stored} rows stored / {polls} polled")

            elapsed = time.monotonic() - cycle_start
            time.sleep(max(0.0, interval - elapsed))
    except KeyboardInterrupt:
        logging.info("Stopped by user")
    finally:
        store.close()
        try:
            if client.get_connected():
                client.disconnect()
        except Exception as e:
            logging.debug(f"Disconnect on shutdown ignored: {e}")
        logging.info(f"Done. {stored} rows stored from {polls} polls in {store.directory}")

    return stored


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stream named S7-300 signals to a rolling CSV store.")
    p.add_argument("--ip", default=DEFAULT_IP, help="PLC IP address")
    p.add_argument("--rack", type=int, default=DEFAULT_RACK)
    p.add_argument("--slot", type=int, default=DEFAULT_SLOT)
    p.add_argument("--symbols", default=DEFAULT_SYMBOLS, help="symbol table CSV")
    p.add_argument("--dir", dest="directory", default=DEFAULT_DATA_DIR, help="data directory")
    p.add_argument("--interval", type=float, default=1.0, help="poll interval (s)")
    p.add_argument("--max-mb", type=float, default=100.0, help="total storage cap (MB)")
    p.add_argument("--segment-mb", type=float, default=None, help="size per CSV segment (MB)")
    p.add_argument("--store-all", action="store_true", help="store every poll, not just on change")
    p.add_argument("--heartbeat", type=float, default=0.0, help="also store an unchanged row every N s")
    p.add_argument("--lite", action="store_true", help="numeric only: counters/analog/words, no bits or timers")
    p.add_argument("--all", action="store_true", help="(default behaviour now; kept for compatibility)")
    p.add_argument("--include-bits", action="store_true", help="force M/I/Q bits even under --lite")
    p.add_argument("--include-timers", action="store_true", help="force timers even under --lite")
    p.add_argument("--watch", default="", help="always log these addresses/symbols, e.g. M115.7,M100.7")
    p.add_argument("--names", default="", help="CSV (Address,Header) to override/rename column headers")
    p.add_argument("--max-polls", type=int, default=None, help="stop after N polls (for testing)")
    p.add_argument("--max-fail-streak", type=int, default=60, help="give up after N all-failed cycles")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    watch = normalize_watch(args.watch.split(",")) if args.watch else set()
    overrides = load_name_overrides(args.names) if args.names else None
    lite = args.lite and not args.all  # default logs everything; --lite trims it
    sigs = parse_symbols(args.symbols,
                         include_bits=(not lite) or args.include_bits,
                         include_timers=(not lite) or args.include_timers,
                         watch=watch, name_overrides=overrides)
    logging.info(f"Loaded {len(sigs)} signals from {args.symbols}"
                 + (f" (watching {sorted(watch)})" if watch else ""))
    stream(
        sigs,
        ip=args.ip,
        rack=args.rack,
        slot=args.slot,
        directory=args.directory,
        interval=args.interval,
        max_mb=args.max_mb,
        segment_mb=args.segment_mb,
        store_on_change=not args.store_all,
        heartbeat_s=args.heartbeat,
        max_polls=args.max_polls,
        max_fail_streak=args.max_fail_streak,
    )
