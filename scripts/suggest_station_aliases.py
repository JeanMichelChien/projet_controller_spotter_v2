#!/usr/bin/env python3
"""Suggest and apply station aliases from unmatched Telegram messages."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from rapidfuzz import fuzz, process

from build_station_events import (
    classify_event_scope,
    load_station_reference,
    normalize_text,
)

REVIEW_COLUMNS = [
    "review_status",
    "alias_candidate",
    "suggested_station",
    "support_count",
    "avg_score",
    "best_score",
    "runner_up_station",
    "runner_up_count",
    "example_messages",
]

APPROVED_VALUES = {"approve", "approved", "yes", "y", "true", "1"}

GENERIC_SINGLE_TOKEN_BLOCKLIST = {
    "achtung",
    "ausgestiegen",
    "auto",
    "autokontrolle",
    "beim",
    "bus",
    "kontrolle",
    "kontrolleure",
    "kontrolleur",
    "kontrol",
    "kontis",
    "langstrasse",
    "polizei",
    "raus",
    "station",
    "stadt",
    "tram",
    "uniform",
    "uniformiert",
    "uniformierte",
    "velo",
    "velokontrolle",
    "warten",
    "zivil",
    "zivis",
}


@dataclass
class StationEvidence:
    count: int = 0
    score_sum: float = 0.0
    best_score: int = 0
    examples: List[str] = field(default_factory=list)

    def add(self, score: int, text: str, max_examples: int) -> None:
        self.count += 1
        self.score_sum += score
        if score > self.best_score:
            self.best_score = score
        if text and text not in self.examples and len(self.examples) < max_examples:
            self.examples.append(text)

    @property
    def avg_score(self) -> float:
        if self.count <= 0:
            return 0.0
        return self.score_sum / self.count


def choose_candidates_by_token_len(
    stations_by_len: Dict[int, List[str]],
    token_len: int,
) -> List[str]:
    """Pick candidate normalized station names near the window token length."""
    lengths = [token_len]
    if token_len - 1 > 0:
        lengths.append(token_len - 1)
    if token_len + 1 in stations_by_len:
        lengths.append(token_len + 1)

    candidates: List[str] = []
    for length in lengths:
        candidates.extend(stations_by_len.get(length, []))
    return candidates


def iter_windows(normalized_text: str, max_window_tokens: int) -> Iterable[str]:
    """Yield unique token windows from a normalized message."""
    tokens = normalized_text.split()
    if not tokens:
        return

    seen: Set[str] = set()
    upper = min(max_window_tokens, len(tokens))
    for n_tokens in range(upper, 0, -1):
        for start_idx in range(0, len(tokens) - n_tokens + 1):
            window = " ".join(tokens[start_idx : start_idx + n_tokens]).strip()
            if not window or window in seen:
                continue
            seen.add(window)
            yield window


def should_consider_alias(alias_norm: str) -> bool:
    """Filter out generic or noisy alias candidates."""
    if len(alias_norm) < 4:
        return False
    if not re.search(r"[a-z]", alias_norm):
        return False
    if re.fullmatch(r"\d+(?:er)?", alias_norm):
        return False
    if re.fullmatch(r"(?:s|sn|ir|re)\d{1,3}", alias_norm):
        return False

    parts = alias_norm.split()
    if len(parts) == 1 and parts[0] in GENERIC_SINGLE_TOKEN_BLOCKLIST:
        return False
    return True


def suggest_aliases(
    unmatched_csv: Path,
    stations_csv: Path,
    output_review_csv: Path,
    min_score: int,
    min_support: int,
    max_window_tokens: int,
    max_examples: int,
) -> Tuple[int, int]:
    """Generate alias suggestions from unmatched messages."""
    station_entries, alias_entries = load_station_reference(stations_csv)
    station_lookup = {entry.normalized: entry for entry in station_entries}
    stations_by_len: Dict[int, List[str]] = {}
    for entry in station_entries:
        stations_by_len.setdefault(entry.token_len, []).append(entry.normalized)

    existing_aliases = {alias for alias, _station in alias_entries}
    existing_station_norms = set(station_lookup.keys())

    # alias_norm -> station_name -> evidence
    evidence: Dict[str, Dict[str, StationEvidence]] = defaultdict(dict)

    with unmatched_csv.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            text = (row.get("text") or "").strip()
            if not text:
                continue

            row_scope = normalize_text(row.get("event_scope") or "")
            if row_scope == "non transport":
                continue

            normalized_text = normalize_text(text)
            if not normalized_text:
                continue

            # If this file predates event_scope column, keep a conservative transport filter.
            if not row_scope and classify_event_scope(normalized_text) == "non_transport":
                continue

            # Avoid counting same alias multiple times within one message.
            per_message_best: Dict[Tuple[str, str], int] = {}

            for window in iter_windows(normalized_text, max_window_tokens=max_window_tokens):
                if not should_consider_alias(window):
                    continue
                if window in existing_aliases or window in existing_station_norms:
                    continue

                candidates = choose_candidates_by_token_len(stations_by_len, len(window.split()))
                if not candidates:
                    continue

                match = process.extractOne(
                    window,
                    candidates,
                    scorer=fuzz.ratio,
                    score_cutoff=min_score,
                )
                if not match:
                    continue

                station_norm, score, _ = match
                station_name = station_lookup[station_norm].name
                key = (window, station_name)

                best_seen = per_message_best.get(key, 0)
                if int(score) > best_seen:
                    per_message_best[key] = int(score)

            for (alias_norm, station_name), score in per_message_best.items():
                by_station = evidence[alias_norm]
                stats = by_station.setdefault(station_name, StationEvidence())
                stats.add(score=score, text=text, max_examples=max_examples)

    suggestions: List[dict] = []
    for alias_norm, by_station in evidence.items():
        ranked = sorted(
            by_station.items(),
            key=lambda item: (-item[1].count, -item[1].avg_score, item[0]),
        )
        if not ranked:
            continue

        top_station, top_stats = ranked[0]
        runner_up_station = ""
        runner_up_count = 0
        if len(ranked) > 1:
            runner_up_station, runner_up_stats = ranked[1]
            runner_up_count = runner_up_stats.count

        if top_stats.count < min_support:
            continue
        if runner_up_count > 0 and top_stats.count < (2 * runner_up_count):
            continue

        example_messages = " || ".join(top_stats.examples)
        suggestions.append(
            {
                "review_status": "",
                "alias_candidate": alias_norm,
                "suggested_station": top_station,
                "support_count": top_stats.count,
                "avg_score": f"{top_stats.avg_score:.1f}",
                "best_score": top_stats.best_score,
                "runner_up_station": runner_up_station,
                "runner_up_count": runner_up_count,
                "example_messages": example_messages,
            }
        )

    suggestions.sort(
        key=lambda row: (
            -int(row["support_count"]),
            -float(row["avg_score"]),
            row["alias_candidate"],
        )
    )

    output_review_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_review_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(suggestions)

    return len(suggestions), len(evidence)


def apply_reviewed_aliases(
    stations_csv: Path,
    review_csv: Path,
    output_csv: Optional[Path],
) -> Tuple[int, int, int, int]:
    """Apply approved alias rows from review CSV to stations CSV."""
    with stations_csv.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError(f"Missing headers in {stations_csv}")
        rows = list(reader)

    if "Aliases" not in fieldnames:
        raise ValueError("Stations CSV must include an 'Aliases' column.")
    if "Station" not in fieldnames:
        raise ValueError("Stations CSV must include a 'Station' column.")

    station_idx_by_norm: Dict[str, int] = {}
    aliases_by_station_idx: Dict[int, Set[str]] = {}
    alias_order_by_station_idx: Dict[int, List[str]] = {}
    alias_owner: Dict[str, str] = {}
    touched_station_indices: Set[int] = set()

    for idx, row in enumerate(rows):
        station_name = (row.get("Station") or "").strip()
        if not station_name:
            continue
        station_norm = normalize_text(station_name)
        if not station_norm:
            continue
        station_idx_by_norm[station_norm] = idx

        raw_aliases = (row.get("Aliases") or "").strip()
        alias_order: List[str] = []
        aliases: Set[str] = set()
        for chunk in raw_aliases.split("|"):
            raw_alias = chunk.strip()
            if not raw_alias:
                continue
            alias_norm = normalize_text(raw_alias)
            if not alias_norm or alias_norm in aliases:
                continue
            aliases.add(alias_norm)
            alias_order.append(raw_alias)

        aliases_by_station_idx[idx] = aliases
        alias_order_by_station_idx[idx] = alias_order
        for alias in aliases:
            alias_owner[alias] = station_norm

    approved = 0
    applied = 0
    already_present = 0
    conflicts = 0

    with review_csv.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            status = normalize_text(row.get("review_status") or "")
            if status not in APPROVED_VALUES:
                continue
            approved += 1

            alias_norm = normalize_text(row.get("alias_candidate") or "")
            station_norm = normalize_text(row.get("suggested_station") or "")
            if not alias_norm or not station_norm:
                conflicts += 1
                continue
            if alias_norm in station_idx_by_norm:
                conflicts += 1
                continue
            if station_norm not in station_idx_by_norm:
                conflicts += 1
                continue

            owner = alias_owner.get(alias_norm)
            if owner == station_norm:
                already_present += 1
                continue
            if owner is not None and owner != station_norm:
                conflicts += 1
                continue

            station_idx = station_idx_by_norm[station_norm]
            aliases = aliases_by_station_idx.setdefault(station_idx, set())
            if alias_norm in aliases:
                already_present += 1
                continue

            aliases.add(alias_norm)
            alias_order_by_station_idx.setdefault(station_idx, []).append(alias_norm)
            alias_owner[alias_norm] = station_norm
            touched_station_indices.add(station_idx)
            applied += 1

    for idx in touched_station_indices:
        rows[idx]["Aliases"] = "|".join(alias_order_by_station_idx.get(idx, []))

    destination = output_csv if output_csv is not None else stations_csv
    with destination.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return approved, applied, already_present, conflicts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Suggest and apply station aliases from unmatched messages."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    suggest = subparsers.add_parser(
        "suggest",
        help="Generate a review CSV with suggested aliases from unmatched messages.",
    )
    suggest.add_argument(
        "--unmatched",
        type=Path,
        default=Path("exports/unmatched_messages.csv"),
        help="Unmatched messages CSV generated by build_station_events.py.",
    )
    suggest.add_argument(
        "--stations",
        type=Path,
        default=Path("data/zurich_stations.csv"),
        help="Stations CSV with canonical names and aliases.",
    )
    suggest.add_argument(
        "--output",
        type=Path,
        default=Path("exports/alias_suggestions_review.csv"),
        help="Output review CSV path.",
    )
    suggest.add_argument(
        "--min-score",
        type=int,
        default=90,
        help="Minimum RapidFuzz score to consider a candidate alias.",
    )
    suggest.add_argument(
        "--min-support",
        type=int,
        default=3,
        help="Minimum unmatched-message support required per suggested alias.",
    )
    suggest.add_argument(
        "--max-window-tokens",
        type=int,
        default=4,
        help="Maximum token-window size to scan inside each unmatched message.",
    )
    suggest.add_argument(
        "--max-examples",
        type=int,
        default=3,
        help="Max example messages stored per suggestion row.",
    )

    apply = subparsers.add_parser(
        "apply",
        help="Apply approved aliases from a reviewed suggestion CSV.",
    )
    apply.add_argument(
        "--stations",
        type=Path,
        default=Path("data/zurich_stations.csv"),
        help="Stations CSV to update.",
    )
    apply.add_argument(
        "--review",
        type=Path,
        default=Path("exports/alias_suggestions_review.csv"),
        help="Review CSV containing review_status values.",
    )
    apply.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path. If omitted, updates --stations in place.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "suggest":
        written, considered = suggest_aliases(
            unmatched_csv=args.unmatched,
            stations_csv=args.stations,
            output_review_csv=args.output,
            min_score=args.min_score,
            min_support=args.min_support,
            max_window_tokens=args.max_window_tokens,
            max_examples=args.max_examples,
        )
        print(f"Candidate aliases considered: {considered}")
        print(f"Suggestions written: {written}")
        print(f"Review CSV: {args.output}")
        return

    approved, applied, already_present, conflicts = apply_reviewed_aliases(
        stations_csv=args.stations,
        review_csv=args.review,
        output_csv=args.output,
    )
    destination = args.output if args.output is not None else args.stations
    print(f"Approved rows processed: {approved}")
    print(f"Aliases applied: {applied}")
    print(f"Already present: {already_present}")
    print(f"Conflicts/skipped: {conflicts}")
    print(f"Updated stations CSV: {destination}")


if __name__ == "__main__":
    main()
