#!/usr/bin/env python3
"""Build a station-level event table from Telegram JSONL messages."""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
from zoneinfo import ZoneInfo

from rapidfuzz import fuzz, process

# Stable output schema consumed by the Streamlit app.
OUTPUT_COLUMNS = [
    "sent_at_utc",
    "sent_at_ch",
    "weekday_ch",
    "hour_ch",
    "event_scope",
    "station",
    "match_status",
    "match_method",
    "match_score",
    "longitude",
    "latitude",
]

UNMATCHED_COLUMNS = [
    "sent_at_utc",
    "sent_at_ch",
    "weekday_ch",
    "hour_ch",
    "event_scope",
    "match_method",
    "text",
]

WEEKDAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

TRANSPORT_KEYWORDS = {
    "bahnhof",
    "billet",
    "bus",
    "gleis",
    "haltestelle",
    "kontrolleur",
    "linie",
    "perron",
    "sbb",
    "station",
    "ticket",
    "tram",
    "vbz",
    "zivis",
    "zug",
    "zvv",
}

NON_TRANSPORT_KEYWORDS = {
    "ampel",
    "auto",
    "autokontrolle",
    "fahrrad",
    "fahrzeug",
    "fahrzeugkontrolle",
    "kreuzung",
    "parkplatz",
    "strasse",
    "strass",
    "tunnel",
    "velo",
    "velokontrolle",
    "verkehr",
    "verkehrskontrolle",
}


@dataclass(frozen=True)
class StationEntry:
    """Canonical station record used during matching."""

    name: str
    normalized: str
    longitude: float
    latitude: float

    @property
    def char_len(self) -> int:
        return len(self.normalized)

    @property
    def token_len(self) -> int:
        return len(self.normalized.split())


def normalize_text(text: str) -> str:
    """Normalize text to a comparable token form for exact/fuzzy matching."""
    text = text.lower().replace("ß", "ss")
    text = "".join(
        char
        for char in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(char)
    )
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_iso8601_utc(value: str) -> datetime:
    """Parse ISO8601 timestamps from JSONL and enforce UTC timezone."""
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return dt.astimezone(timezone.utc)


def parse_aliases(raw_aliases: str) -> Set[str]:
    """Parse pipe-separated aliases from the stations CSV row."""
    aliases: Set[str] = set()
    for chunk in raw_aliases.split("|"):
        normalized = normalize_text(chunk)
        if normalized:
            aliases.add(normalized)
    return aliases


def load_station_reference(
    stations_csv: Path,
) -> Tuple[List[StationEntry], List[Tuple[str, StationEntry]]]:
    """
    Load canonical stations + alias mappings from one CSV.

    Returns:
    - station entries (canonical names and coordinates)
    - alias tuples: (normalized alias, StationEntry)
    """
    by_station_name: Dict[str, Tuple[float, float]] = {}
    aliases_by_station: Dict[str, Set[str]] = {}

    with stations_csv.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            station = (row.get("Station") or "").strip()
            lon_raw = (row.get("Longitude") or "").strip()
            lat_raw = (row.get("Latitude") or "").strip()
            aliases_raw = (row.get("Aliases") or "").strip()

            if not station:
                continue
            if lon_raw in {"", "#N/A"} or lat_raw in {"", "#N/A"}:
                continue

            lon = float(lon_raw)
            lat = float(lat_raw)
            if not (5.0 <= lon <= 12.0 and 45.0 <= lat <= 49.5):
                # Drop malformed coordinates from the source file.
                continue

            previous = by_station_name.get(station)
            if previous and previous != (lon, lat):
                raise ValueError(
                    f"Station '{station}' has multiple coordinate pairs: {previous} vs {(lon, lat)}"
                )
            by_station_name[station] = (lon, lat)
            aliases_by_station.setdefault(station, set()).update(parse_aliases(aliases_raw))

    entries: List[StationEntry] = []
    for station, (lon, lat) in by_station_name.items():
        normalized = normalize_text(station)
        if not normalized:
            continue
        entries.append(
            StationEntry(
                name=station,
                normalized=normalized,
                longitude=lon,
                latitude=lat,
            )
        )

    entries.sort(key=lambda item: (-item.char_len, item.name))

    # Build a unique alias -> station map and reject conflicts early.
    by_normalized_station = {entry.normalized: entry for entry in entries}
    aliases: Dict[str, StationEntry] = {}
    for station_name, station_aliases in aliases_by_station.items():
        station_normalized = normalize_text(station_name)
        station_entry = by_normalized_station.get(station_normalized)
        if station_entry is None:
            continue

        for alias_norm in station_aliases:
            if alias_norm == station_normalized:
                continue
            existing = aliases.get(alias_norm)
            if existing and existing.name != station_entry.name:
                raise ValueError(
                    f"Alias '{alias_norm}' maps to multiple stations: "
                    f"'{existing.name}' and '{station_entry.name}'"
                )
            aliases[alias_norm] = station_entry

    alias_entries = sorted(
        aliases.items(),
        key=lambda pair: (-len(pair[0]), pair[1].name),
    )

    return entries, alias_entries


def find_exact_station_match(
    normalized_text: str,
    stations: Iterable[StationEntry],
) -> Optional[Tuple[StationEntry, int]]:
    """Find best exact station phrase match (longest phrase, earliest position)."""
    if not normalized_text:
        return None

    padded_text = f" {normalized_text} "
    best: Optional[Tuple[StationEntry, int]] = None

    for entry in stations:
        phrase = f" {entry.normalized} "
        found_at = padded_text.find(phrase)
        if found_at < 0:
            continue

        start_index = max(found_at - 1, 0)

        if best is None:
            best = (entry, start_index)
            continue

        best_entry, best_start = best
        if entry.char_len > best_entry.char_len:
            best = (entry, start_index)
            continue
        if entry.char_len == best_entry.char_len:
            if start_index < best_start:
                best = (entry, start_index)
                continue
            if start_index == best_start and entry.name < best_entry.name:
                best = (entry, start_index)

    return best


def find_fuzzy_station_match(
    normalized_text: str,
    stations_by_len: Dict[int, List[str]],
    station_lookup: Dict[str, StationEntry],
    score_cutoff: int,
) -> Optional[Tuple[StationEntry, int]]:
    """Conservative fuzzy fallback over token windows near station token lengths."""
    tokens = normalized_text.split()
    if not tokens:
        return None

    best_station: Optional[StationEntry] = None
    best_score = -1
    best_start = 10**9

    max_station_tokens = max(stations_by_len.keys(), default=1)
    n_max = min(len(tokens), max_station_tokens + 1)

    for n_tokens in range(n_max, 0, -1):
        candidate_lens = [n_tokens]
        if n_tokens - 1 > 0:
            candidate_lens.append(n_tokens - 1)
        if n_tokens + 1 <= max_station_tokens:
            candidate_lens.append(n_tokens + 1)

        candidates: List[str] = []
        for candidate_len in candidate_lens:
            candidates.extend(stations_by_len.get(candidate_len, []))
        if not candidates:
            continue

        for start_idx in range(0, len(tokens) - n_tokens + 1):
            window = " ".join(tokens[start_idx : start_idx + n_tokens])
            # Very short windows are too noisy for fuzzy matching.
            if n_tokens == 1 and len(window) < 6:
                continue
            if len(window) < 5:
                continue

            match = process.extractOne(
                window,
                candidates,
                scorer=fuzz.ratio,
                score_cutoff=score_cutoff,
            )
            if not match:
                continue

            matched_norm, score, _ = match
            station = station_lookup[matched_norm]

            is_better = False
            if score > best_score:
                is_better = True
            elif score == best_score and best_station is not None:
                if station.char_len > best_station.char_len:
                    is_better = True
                elif station.char_len == best_station.char_len and start_idx < best_start:
                    is_better = True
                elif (
                    station.char_len == best_station.char_len
                    and start_idx == best_start
                    and station.name < best_station.name
                ):
                    is_better = True

            if is_better:
                best_station = station
                best_score = int(round(score))
                best_start = start_idx

    if best_station is None:
        return None

    return best_station, best_score


def find_alias_station_match(
    normalized_text: str,
    alias_entries: Iterable[Tuple[str, StationEntry]],
) -> Optional[Tuple[StationEntry, int]]:
    """Find best alias hit using the same ranking rule as exact matching."""
    if not normalized_text:
        return None

    padded_text = f" {normalized_text} "
    best: Optional[Tuple[StationEntry, int, int]] = None

    for alias_norm, station in alias_entries:
        found_at = padded_text.find(f" {alias_norm} ")
        if found_at < 0:
            continue

        start_index = max(found_at - 1, 0)
        alias_len = len(alias_norm)
        if best is None:
            best = (station, alias_len, start_index)
            continue

        best_station, best_alias_len, best_start = best
        if alias_len > best_alias_len:
            best = (station, alias_len, start_index)
            continue
        if alias_len == best_alias_len:
            if start_index < best_start:
                best = (station, alias_len, start_index)
                continue
            if start_index == best_start and station.name < best_station.name:
                best = (station, alias_len, start_index)

    if best is None:
        return None
    return best[0], best[2]


def classify_event_scope(normalized_text: str) -> str:
    """
    Classify message scope before fuzzy matching.

    Returns one of:
    - transport: likely public transport ticket-control context
    - non_transport: likely road/traffic police context
    - unknown: insufficient evidence
    """
    if not normalized_text:
        return "unknown"

    tokens = normalized_text.split()
    token_set = set(tokens)

    transport_hits = len(token_set & TRANSPORT_KEYWORDS)
    non_transport_hits = len(token_set & NON_TRANSPORT_KEYWORDS)

    # Typical line patterns such as S7, IR13, RE48.
    if re.search(r"\b(?:s|sn|ir|re)\d{1,3}\b", normalized_text):
        transport_hits += 2
    # Swiss colloquial line style: "32er", "83er", etc.
    if re.search(r"\b\d{1,3}er\b", normalized_text):
        transport_hits += 1

    # "polizei" alone is ambiguous; weight it only when no transport hints exist.
    if "polizei" in token_set and transport_hits == 0:
        non_transport_hits += 1

    if transport_hits >= 2 and transport_hits >= non_transport_hits:
        return "transport"
    if non_transport_hits >= 2 and transport_hits == 0:
        return "non_transport"
    if non_transport_hits >= 3 and non_transport_hits > transport_hits + 1:
        return "non_transport"
    return "unknown"


def build_station_events(
    input_jsonl: Path,
    stations_csv: Path,
    output_csv: Path,
    unmatched_output_csv: Optional[Path],
    fuzzy_score_cutoff: int,
) -> Tuple[int, int, int, int]:
    """
    Build the intermediate CSV consumed by the map app.

    Matching order:
    1) exact station name
    2) station alias
    3) fuzzy station name
    4) unmatched
    """
    tz_ch = ZoneInfo("Europe/Zurich")
    station_entries, alias_entries = load_station_reference(stations_csv)

    station_lookup = {entry.normalized: entry for entry in station_entries}
    stations_by_len: Dict[int, List[str]] = {}
    for entry in station_entries:
        stations_by_len.setdefault(entry.token_len, []).append(entry.normalized)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if unmatched_output_csv is not None:
        unmatched_output_csv.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    matched = 0
    unmatched = 0
    non_transport = 0

    with (
        input_jsonl.open("r", encoding="utf-8") as src,
        output_csv.open("w", encoding="utf-8", newline="") as dst,
    ):
        writer = csv.DictWriter(dst, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        unmatched_writer = None
        unmatched_dst = None
        if unmatched_output_csv is not None:
            unmatched_dst = unmatched_output_csv.open("w", encoding="utf-8", newline="")
            unmatched_writer = csv.DictWriter(unmatched_dst, fieldnames=UNMATCHED_COLUMNS)
            unmatched_writer.writeheader()

        try:
            for raw_line in src:
                line = raw_line.strip()
                if not line:
                    continue

                payload = json.loads(line)
                total += 1

                sent_at_utc = parse_iso8601_utc(payload["sent_at"])
                sent_at_ch = sent_at_utc.astimezone(tz_ch)

                text = payload.get("text") or ""
                normalized_text = normalize_text(text)

                # Default state for messages with no usable station match.
                event_scope = "unknown"
                station_name = "unmatched"
                match_status = "unmatched"
                match_method = "none"
                match_score: Optional[int] = None
                longitude = ""
                latitude = ""

                exact = find_exact_station_match(normalized_text, station_entries)
                if exact is not None:
                    station, _ = exact
                    event_scope = "transport"
                    station_name = station.name
                    match_status = "matched"
                    match_method = "exact"
                    match_score = 100
                    longitude = station.longitude
                    latitude = station.latitude
                else:
                    alias = find_alias_station_match(normalized_text, alias_entries)
                    if alias is not None:
                        station, _ = alias
                        event_scope = "transport"
                        station_name = station.name
                        match_status = "matched"
                        match_method = "alias"
                        match_score = 100
                        longitude = station.longitude
                        latitude = station.latitude
                    else:
                        event_scope = classify_event_scope(normalized_text)
                        if event_scope == "non_transport":
                            match_method = "non_transport"
                            non_transport += 1
                        else:
                            fuzzy = find_fuzzy_station_match(
                                normalized_text,
                                stations_by_len,
                                station_lookup,
                                score_cutoff=fuzzy_score_cutoff,
                            )
                            if fuzzy is not None:
                                station, score = fuzzy
                                event_scope = "transport"
                                station_name = station.name
                                match_status = "matched"
                                match_method = "fuzzy"
                                match_score = score
                                longitude = station.longitude
                                latitude = station.latitude

                if match_status == "matched":
                    matched += 1
                else:
                    unmatched += 1

                sent_at_utc_text = (
                    sent_at_utc.replace(microsecond=0).isoformat().replace("+00:00", "Z")
                )
                sent_at_ch_text = sent_at_ch.replace(microsecond=0).isoformat()
                weekday_ch = WEEKDAY_ORDER[sent_at_ch.weekday()]
                hour_ch = sent_at_ch.hour

                writer.writerow(
                    {
                        "sent_at_utc": sent_at_utc_text,
                        "sent_at_ch": sent_at_ch_text,
                        "weekday_ch": weekday_ch,
                        "hour_ch": hour_ch,
                        "event_scope": event_scope,
                        "station": station_name,
                        "match_status": match_status,
                        "match_method": match_method,
                        "match_score": match_score if match_score is not None else "",
                        "longitude": longitude,
                        "latitude": latitude,
                    }
                )

                if unmatched_writer is not None and match_status == "unmatched":
                    unmatched_writer.writerow(
                        {
                            "sent_at_utc": sent_at_utc_text,
                            "sent_at_ch": sent_at_ch_text,
                            "weekday_ch": weekday_ch,
                            "hour_ch": hour_ch,
                            "event_scope": event_scope,
                            "match_method": match_method,
                            "text": text,
                        }
                    )
        finally:
            if unmatched_dst is not None:
                unmatched_dst.close()

    return total, matched, unmatched, non_transport


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract station events from Telegram messages into a clean CSV datasource."
    )
    parser.add_argument(
        "--input",
        default="exports/messages.jsonl",
        type=Path,
        help="Path to Telegram messages JSONL file.",
    )
    parser.add_argument(
        "--stations",
        default="data/zurich_stations.csv",
        type=Path,
        help="Path to Zurich stations reference CSV.",
    )
    parser.add_argument(
        "--output",
        default="data/station_events.csv",
        type=Path,
        help="Output path for extracted station events CSV.",
    )
    parser.add_argument(
        "--unmatched-output",
        default="exports/unmatched_messages.csv",
        type=Path,
        help="Optional CSV path for unmatched messages with raw text (set empty to disable).",
    )
    parser.add_argument(
        "--fuzzy-threshold",
        default=90,
        type=int,
        help="RapidFuzz score cutoff (0-100).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    unmatched_output = args.unmatched_output
    if str(unmatched_output).strip() == "":
        unmatched_output = None

    total, matched, unmatched, non_transport = build_station_events(
        input_jsonl=args.input,
        stations_csv=args.stations,
        output_csv=args.output,
        unmatched_output_csv=unmatched_output,
        fuzzy_score_cutoff=args.fuzzy_threshold,
    )

    print(f"Input messages processed: {total}")
    print(f"Matched stations: {matched}")
    print(f"Unmatched: {unmatched}")
    print(f"Marked non-transport (matching skipped): {non_transport}")
    print(f"Output written: {args.output}")
    if unmatched_output is not None:
        print(f"Unmatched CSV written: {unmatched_output}")


if __name__ == "__main__":
    main()
