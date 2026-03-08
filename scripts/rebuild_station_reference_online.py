#!/usr/bin/env python3
"""Rebuild data/zurich_stations.csv from official online open data sources."""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import unicodedata
import urllib.parse
import urllib.request
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from urllib.error import HTTPError, URLError
from typing import Dict, Iterable, List, Optional, Set, Tuple

# Stable output schema consumed by build_station_events.py.
OUTPUT_COLUMNS = ["Station", "Type", "line", "Longitude", "Latitude", "Aliases"]

# Defaults chosen for Zurich-region scope and official CKAN API endpoints.
DEFAULT_CITY_API_BASE = "https://data.stadt-zuerich.ch/api/3/action"
DEFAULT_CITY_PACKAGE = "ktzh_haltestellen_des_oeffentlichen_verkehrs___ogd_"
DEFAULT_GTFS_API_BASE = "https://ckan.opendata.swiss/api/3/action"
DEFAULT_GTFS_PACKAGE = "timetable-2026-gtfs2020"

# Zurich / Canton area default bounding box to keep the table relevant.
DEFAULT_MIN_LAT = 47.20
DEFAULT_MAX_LAT = 47.72
DEFAULT_MIN_LON = 8.30
DEFAULT_MAX_LON = 8.95


@dataclass
class OGDStation:
    """Station candidate parsed from Zurich OGD resource."""

    name: str
    normalized: str
    longitude: Optional[float]
    latitude: Optional[float]
    source_type: str = "station"
    source_lines: Set[str] = field(default_factory=set)


@dataclass
class GTFSStop:
    """Raw GTFS stop record with optional parent stop hierarchy."""

    stop_id: str
    name: str
    normalized: str
    longitude: float
    latitude: float
    parent_station: Optional[str]


@dataclass
class StationAggregate:
    """Canonical station row that will be written to the final CSV."""

    station: str
    normalized: str
    longitude: Optional[float]
    latitude: Optional[float]
    station_type: str = "station"
    lines: Set[str] = field(default_factory=set)
    aliases: Set[str] = field(default_factory=set)
    _coord_samples: List[Tuple[float, float]] = field(default_factory=list)
    _gtfs_names: Counter[str] = field(default_factory=Counter)
    _route_type_counts: Counter[int] = field(default_factory=Counter)

    def add_coordinate_sample(self, lon: float, lat: float) -> None:
        """Store coordinates for later near-equal clustering."""
        self._coord_samples.append((lon, lat))

    def choose_final_coordinate(self) -> None:
        """Pick one coordinate pair using deterministic near-equal tie-breaking."""
        if self.longitude is not None and self.latitude is not None:
            return
        if not self._coord_samples:
            return

        # Near-equal coordinates are merged by 4-decimal buckets (~11m lat precision).
        buckets: Counter[Tuple[float, float]] = Counter(
            (round(lon, 4), round(lat, 4)) for lon, lat in self._coord_samples
        )
        best_bucket = sorted(
            buckets.items(),
            key=lambda item: (-item[1], item[0][1], item[0][0]),
        )[0][0]
        self.longitude, self.latitude = best_bucket

    def choose_final_name(self) -> None:
        """Fallback to the most frequent GTFS name if OGD did not provide one."""
        if self.station:
            return
        if not self._gtfs_names:
            self.station = self.normalized
            return
        self.station = sorted(
            self._gtfs_names.items(),
            key=lambda item: (-item[1], len(item[0]), item[0].lower()),
        )[0][0]

    def choose_final_type(self) -> None:
        """Classify station type from dominant GTFS route_type if needed."""
        if self.station_type != "station":
            return
        if not self._route_type_counts:
            return
        route_type = sorted(
            self._route_type_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[0][0]
        self.station_type = route_type_to_label(route_type)


def normalize_text(text: str) -> str:
    """Normalize text for stable matching and joins."""
    text = text.lower().replace("ß", "ss")
    text = "".join(
        ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)
    )
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def natural_sort_key(value: str) -> List[object]:
    """Sort helper so lines like 2, 10, S8 appear in a human-friendly order."""
    chunks = re.split(r"(\d+)", value.strip().lower())
    key: List[object] = []
    for chunk in chunks:
        if chunk.isdigit():
            key.append(int(chunk))
        else:
            key.append(chunk)
    return key


def parse_float(value: str) -> Optional[float]:
    """Parse float safely from CSV text fields."""
    if value is None:
        return None
    raw = str(value).strip()
    if raw in {"", "#N/A", "nan", "None", "NULL", "null"}:
        return None
    raw = raw.replace(",", ".")
    try:
        return float(raw)
    except ValueError:
        return None


def make_request(url: str) -> urllib.request.Request:
    """Build HTTP request with explicit headers for stricter endpoints."""
    return urllib.request.Request(
        url,
        headers={
            "User-Agent": "zurich-controller-spotter/1.0 (+https://github.com)",
            "Accept": "application/json, text/plain, */*",
        },
    )


def fetch_json(url: str, timeout: int = 60) -> dict:
    """GET JSON payload from a remote URL."""
    with urllib.request.urlopen(make_request(url), timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_bytes(url: str, timeout: int = 120) -> bytes:
    """GET raw bytes from a remote URL."""
    with urllib.request.urlopen(make_request(url), timeout=timeout) as response:
        return response.read()


def package_show_url(api_base: str, package_id: str) -> str:
    """Build CKAN package_show URL for a given package id."""
    encoded_id = urllib.parse.quote(package_id, safe="")
    return f"{api_base.rstrip('/')}/package_show?id={encoded_id}"


def package_id_variants(package_id: str) -> List[str]:
    """Return candidate ids for English/German naming variants."""
    variants = [package_id]
    if package_id.startswith("timetable-"):
        variants.append("fahrplan-" + package_id[len("timetable-") :])
    if package_id.startswith("fahrplan-"):
        variants.append("timetable-" + package_id[len("fahrplan-") :])

    seen: Set[str] = set()
    ordered: List[str] = []
    for item in variants:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def fetch_package_with_fallback(api_bases: List[str], package_id: str) -> Tuple[dict, str, str]:
    """Fetch package metadata trying multiple API bases and id variants."""
    errors: List[str] = []
    for api_base in api_bases:
        for candidate in package_id_variants(package_id):
            url = package_show_url(api_base, candidate)
            try:
                payload = fetch_json(url)
            except (HTTPError, URLError, TimeoutError, ValueError) as exc:
                errors.append(f"{api_base} :: {candidate} -> {exc}")
                continue
            if payload.get("success"):
                return payload, api_base, candidate
            errors.append(f"{api_base} :: {candidate} -> success=false")

    raise ValueError(
        "Unable to resolve package metadata. Tried:\n- " + "\n- ".join(errors)
    )


def text_from_ckan_field(value: object) -> str:
    """Normalize CKAN multilingual fields to plain text."""
    if isinstance(value, dict):
        parts: List[str] = []
        for item in value.values():
            if item is None:
                continue
            text = str(item).strip()
            if text:
                parts.append(text)
        return " ".join(parts)
    if value is None:
        return ""
    return str(value)


def choose_ckan_resource(resources: List[dict], kind: str) -> str:
    """Choose best matching CKAN resource URL for CSV (OGD) or ZIP (GTFS)."""
    scored: List[Tuple[int, int, str]] = []
    for resource in resources:
        url = (resource.get("url") or "").strip()
        if not url:
            continue
        format_value = text_from_ckan_field(resource.get("format")).strip().lower()
        name = text_from_ckan_field(resource.get("name")).strip().lower()
        desc = text_from_ckan_field(resource.get("description")).strip().lower()
        haystack = " ".join([format_value, name, desc, url.lower()])

        score = 0
        if kind == "ogd_csv":
            if "csv" in format_value:
                score += 60
            if url.lower().endswith(".csv"):
                score += 40
            if "haltest" in haystack:
                score += 15
            if "verkehr" in haystack:
                score += 5
        elif kind == "gtfs_zip":
            if "zip" in format_value:
                score += 60
            if url.lower().endswith(".zip"):
                score += 40
            if "gtfs" in haystack:
                score += 20
            if "fahrplan" in haystack or "timetable" in haystack:
                score += 5

        # Prefer newest dated GTFS snapshots if date appears in resource metadata.
        date_score = 0
        if kind == "gtfs_zip":
            date_hits = re.findall(r"(20\d{6}|\d{4}-\d{2}-\d{2})", haystack)
            if date_hits:
                normalized_dates = [hit.replace("-", "") for hit in date_hits]
                date_score = max(int(value) for value in normalized_dates)

        if score > 0:
            scored.append((score, date_score, url))

    if not scored:
        raise ValueError(f"Could not find a suitable {kind} resource in package metadata.")

    return sorted(scored, key=lambda item: (-item[0], -item[1], item[2]))[0][2]


def detect_csv_dialect(sample: str) -> csv.Dialect:
    """Infer delimiter/quoting for third-party CSV payloads."""
    try:
        return csv.Sniffer().sniff(sample, delimiters=";,|\t,")
    except csv.Error:
        # Fallback to semicolon first, common for Swiss open-data exports.
        class _Fallback(csv.Dialect):
            delimiter = ";"
            quotechar = '"'
            doublequote = True
            skipinitialspace = False
            lineterminator = "\n"
            quoting = csv.QUOTE_MINIMAL

        return _Fallback()


def decoded_text(data: bytes) -> str:
    """Decode bytes with robust fallback for open-data text files."""
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    # Final fallback keeps process alive even for mixed encodings.
    return data.decode("utf-8", errors="replace")


def normalize_header(value: str) -> str:
    """Normalize header names to simplify fuzzy column detection."""
    return normalize_text(value).replace(" ", "")


def pick_header(headers: Iterable[str], candidates: Set[str]) -> Optional[str]:
    """Pick first header matching a candidate set after normalization."""
    for header in headers:
        if normalize_header(header) in candidates:
            return header
    return None


def parse_geo_point(value: str) -> Tuple[Optional[float], Optional[float]]:
    """Parse single geopoint field to (lon, lat) when present."""
    if not value:
        return None, None
    parts = [part.strip() for part in re.split(r"[,\s]+", value) if part.strip()]
    if len(parts) < 2:
        return None, None
    lat = parse_float(parts[0])
    lon = parse_float(parts[1])
    return lon, lat


def parse_ogd_stations(csv_bytes: bytes) -> Dict[str, OGDStation]:
    """Parse OGD station CSV into normalized station records."""
    text = decoded_text(csv_bytes)
    dialect = detect_csv_dialect(text[:5000])
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    if reader.fieldnames is None:
        raise ValueError("OGD CSV has no headers.")

    headers = [header.strip() for header in reader.fieldnames]
    name_header = pick_header(
        headers,
        {
            "station",
            "stopname",
            "name",
            "haltestelle",
            "haltestellenname",
            "bezeichnung",
        },
    )
    lon_header = pick_header(
        headers,
        {"longitude", "lon", "xwgs84", "wgs84lon", "stoplon"},
    )
    lat_header = pick_header(
        headers,
        {"latitude", "lat", "ywgs84", "wgs84lat", "stoplat"},
    )
    point_header = pick_header(headers, {"geopoint2d", "geo_point_2d", "wgs84"})
    type_header = pick_header(
        headers,
        {"type", "typ", "haltestellentyp", "verkehrsmittel", "transportmittel"},
    )
    line_header = pick_header(headers, {"linie", "linien", "line", "routes"})

    if not name_header:
        raise ValueError(f"Could not detect station name column in OGD CSV headers: {headers}")

    stations: Dict[str, OGDStation] = {}
    for row in reader:
        raw_name = (row.get(name_header) or "").strip()
        if not raw_name:
            continue
        normalized = normalize_text(raw_name)
        if not normalized:
            continue

        lon = parse_float(row.get(lon_header, "")) if lon_header else None
        lat = parse_float(row.get(lat_header, "")) if lat_header else None
        if (lon is None or lat is None) and point_header:
            p_lon, p_lat = parse_geo_point(row.get(point_header, ""))
            lon = lon if lon is not None else p_lon
            lat = lat if lat is not None else p_lat

        source_type = "station"
        if type_header:
            source_type = normalize_source_type(row.get(type_header, ""))

        source_lines: Set[str] = set()
        if line_header:
            raw_lines = row.get(line_header, "")
            for chunk in re.split(r"[|,;/]+", raw_lines):
                line = chunk.strip()
                if line:
                    source_lines.add(line)

        existing = stations.get(normalized)
        if existing is None:
            stations[normalized] = OGDStation(
                name=raw_name,
                normalized=normalized,
                longitude=lon,
                latitude=lat,
                source_type=source_type,
                source_lines=source_lines,
            )
            continue

        # Deterministic tie-break: keep lexicographically smaller display name.
        if raw_name.lower() < existing.name.lower():
            existing.name = raw_name

        if existing.longitude is None and lon is not None:
            existing.longitude = lon
        if existing.latitude is None and lat is not None:
            existing.latitude = lat
        existing.source_lines.update(source_lines)
        if existing.source_type == "station" and source_type != "station":
            existing.source_type = source_type

    return stations


def parse_ogd_wfs_geojson(payload: bytes) -> Dict[str, OGDStation]:
    """Parse Zurich OGD WFS GeoJSON response into station records."""
    obj = json.loads(decoded_text(payload))
    features = obj.get("features", [])
    if not isinstance(features, list):
        raise ValueError("Invalid OGD WFS GeoJSON: missing features list.")

    stations: Dict[str, OGDStation] = {}
    for feature in features:
        if not isinstance(feature, dict):
            continue
        props = feature.get("properties") or {}
        geom = feature.get("geometry") or {}
        if not isinstance(props, dict) or not isinstance(geom, dict):
            continue

        raw_name = (
            str(props.get("chstname") or props.get("stop_name") or props.get("name") or "").strip()
        )
        if not raw_name:
            continue
        normalized = normalize_text(raw_name)
        if not normalized:
            continue

        lon: Optional[float] = None
        lat: Optional[float] = None
        coords = geom.get("coordinates")
        if isinstance(coords, list) and len(coords) >= 2:
            lon = parse_float(coords[0])
            lat = parse_float(coords[1])

        source_type = normalize_source_type(str(props.get("vtyp") or props.get("hsttyp") or ""))
        source_lines: Set[str] = set()
        raw_lines = str(props.get("linien") or "")
        for chunk in re.split(r"[|,;/]+", raw_lines):
            line = chunk.strip()
            if line:
                source_lines.add(line)

        existing = stations.get(normalized)
        if existing is None:
            stations[normalized] = OGDStation(
                name=raw_name,
                normalized=normalized,
                longitude=lon,
                latitude=lat,
                source_type=source_type,
                source_lines=source_lines,
            )
            continue

        if raw_name.lower() < existing.name.lower():
            existing.name = raw_name
        if existing.longitude is None and lon is not None:
            existing.longitude = lon
        if existing.latitude is None and lat is not None:
            existing.latitude = lat
        existing.source_lines.update(source_lines)
        if existing.source_type == "station" and source_type != "station":
            existing.source_type = source_type

    return stations


def build_wfs_geojson_url(wfs_base_url: str) -> str:
    """Build a GetFeature URL returning GeoJSON with WGS84 coordinates."""
    params = {
        "SERVICE": "WFS",
        "VERSION": "2.0.0",
        "REQUEST": "GetFeature",
        "TYPENAMES": "ms:haltestellen",
        "OUTPUTFORMAT": "application/json; subtype=geojson",
        "SRSNAME": "EPSG:4326",
    }
    return f"{wfs_base_url}?{urllib.parse.urlencode(params)}"


def load_ogd_stations_from_resources(resources: List[dict]) -> Tuple[Dict[str, OGDStation], str]:
    """Load OGD stations, preferring CSV and falling back to WFS GeoJSON."""
    csv_url = choose_ckan_resource(resources, kind="ogd_csv")
    csv_payload = fetch_bytes(csv_url)
    try:
        stations = parse_ogd_stations(csv_payload)
        if stations:
            return stations, csv_url
    except Exception:
        # CSV endpoint can return an HTML landing page; continue with WFS fallback.
        pass

    wfs_url: Optional[str] = None
    for resource in resources:
        fmt = text_from_ckan_field(resource.get("format")).strip().lower()
        name = text_from_ckan_field(resource.get("name")).strip().lower()
        url = text_from_ckan_field(resource.get("url")).strip()
        if not url:
            continue
        if "wfs" in fmt and "haltestellen" in name:
            wfs_url = url
            break
        if "haltestellenzhwfs" in url.lower():
            wfs_url = url
            break

    if not wfs_url:
        raise ValueError("Could not find OGD WFS resource for station fallback.")

    wfs_geojson_url = build_wfs_geojson_url(wfs_url)
    stations = parse_ogd_wfs_geojson(fetch_bytes(wfs_geojson_url))
    if not stations:
        raise ValueError("OGD WFS fallback returned zero stations.")
    return stations, wfs_geojson_url


def normalize_source_type(raw_type: str) -> str:
    """Normalize source transport type to app-compatible labels."""
    value = normalize_text(raw_type)
    if not value:
        return "station"
    if "tram" in value:
        return "tram station"
    if "bus" in value:
        return "bus station"
    if "zug" in value or "train" in value or "rail" in value or "bahn" in value:
        return "train station"
    return "station"


def read_zip_csv_rows(zf: zipfile.ZipFile, filename: str) -> Iterable[dict]:
    """Yield dict rows from a CSV file inside a GTFS ZIP archive."""
    with zf.open(filename) as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
        reader = csv.DictReader(text)
        for row in reader:
            yield row


def route_type_to_label(route_type: int) -> str:
    """Map GTFS route_type values to current CSV type labels."""
    if route_type == 0:
        return "tram station"
    if route_type in {1, 2, 100, 101, 102, 103, 106, 107, 109}:
        return "train station"
    if route_type in {3, 700, 701, 702, 704, 705, 706, 707, 708}:
        return "bus station"
    return "station"


def load_gtfs_stops_and_routes(
    gtfs_zip: bytes,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    ogd_names: Set[str],
) -> Tuple[Dict[str, GTFSStop], Dict[str, Set[str]], Dict[str, Counter[int]]]:
    """Load scoped GTFS stops and derive served route short names per canonical stop."""
    with zipfile.ZipFile(io.BytesIO(gtfs_zip)) as zf:
        expected = {"stops.txt", "stop_times.txt", "trips.txt", "routes.txt"}
        names = set(zf.namelist())
        missing = sorted(expected - names)
        if missing:
            raise ValueError(f"GTFS ZIP is missing required files: {missing}")

        # route_id -> (short_name, route_type)
        route_meta: Dict[str, Tuple[str, int]] = {}
        for row in read_zip_csv_rows(zf, "routes.txt"):
            route_id = (row.get("route_id") or "").strip()
            if not route_id:
                continue
            short_name = (
                (row.get("route_short_name") or "").strip()
                or (row.get("route_long_name") or "").strip()
            )
            route_type_raw = (row.get("route_type") or "").strip()
            try:
                route_type = int(route_type_raw)
            except ValueError:
                route_type = -1
            route_meta[route_id] = (short_name, route_type)

        # trip_id -> route_id
        trip_to_route: Dict[str, str] = {}
        for row in read_zip_csv_rows(zf, "trips.txt"):
            trip_id = (row.get("trip_id") or "").strip()
            route_id = (row.get("route_id") or "").strip()
            if trip_id and route_id:
                trip_to_route[trip_id] = route_id

        # Parse all stops first so child/platform stops can be collapsed to parent stations.
        all_stops: Dict[str, GTFSStop] = {}
        for row in read_zip_csv_rows(zf, "stops.txt"):
            stop_id = (row.get("stop_id") or "").strip()
            raw_name = (row.get("stop_name") or "").strip()
            lon = parse_float(row.get("stop_lon"))
            lat = parse_float(row.get("stop_lat"))
            if not stop_id or not raw_name or lon is None or lat is None:
                continue
            parent_station = (row.get("parent_station") or "").strip() or None
            normalized = normalize_text(raw_name)
            all_stops[stop_id] = GTFSStop(
                stop_id=stop_id,
                name=raw_name,
                normalized=normalized,
                longitude=lon,
                latitude=lat,
                parent_station=parent_station,
            )

        def canonical_stop_id(stop_id: str) -> str:
            stop = all_stops.get(stop_id)
            if stop is None:
                return stop_id
            parent = stop.parent_station
            if parent and parent in all_stops:
                return parent
            return stop_id

        # Determine which canonical stations belong to Zurich scope.
        scoped_canonical_ids: Set[str] = set()
        for stop_id, stop in all_stops.items():
            cid = canonical_stop_id(stop_id)
            canonical = all_stops.get(cid, stop)

            in_bbox = (
                min_lon <= canonical.longitude <= max_lon
                and min_lat <= canonical.latitude <= max_lat
            )
            in_ogd_name_set = canonical.normalized in ogd_names
            if in_bbox or in_ogd_name_set:
                scoped_canonical_ids.add(cid)

        scoped_stops: Dict[str, GTFSStop] = {}
        for stop_id, stop in all_stops.items():
            cid = canonical_stop_id(stop_id)
            if cid in scoped_canonical_ids:
                # Station-level table is keyed by canonical stop ids.
                scoped_stops[cid] = all_stops[cid]

        # stop_id -> canonical_stop_id for fast lookup during stop_times scan.
        scoped_child_to_canonical: Dict[str, str] = {}
        for stop_id in all_stops:
            cid = canonical_stop_id(stop_id)
            if cid in scoped_canonical_ids:
                scoped_child_to_canonical[stop_id] = cid

        # Derive per-station lines and route_type counters.
        stop_lines: Dict[str, Set[str]] = defaultdict(set)
        stop_route_types: Dict[str, Counter[int]] = defaultdict(Counter)

        for row in read_zip_csv_rows(zf, "stop_times.txt"):
            stop_id = (row.get("stop_id") or "").strip()
            trip_id = (row.get("trip_id") or "").strip()
            if not stop_id or not trip_id:
                continue
            canonical_id = scoped_child_to_canonical.get(stop_id)
            if canonical_id is None:
                continue

            route_id = trip_to_route.get(trip_id)
            if route_id is None:
                continue
            short_name, route_type = route_meta.get(route_id, ("", -1))
            if short_name:
                stop_lines[canonical_id].add(short_name)
            if route_type >= 0:
                stop_route_types[canonical_id][route_type] += 1

        return scoped_stops, stop_lines, stop_route_types


def load_existing_aliases(csv_path: Path) -> Dict[str, Set[str]]:
    """Load aliases from current CSV so refresh keeps manual curation."""
    if not csv_path.exists():
        return {}

    aliases_by_station: Dict[str, Set[str]] = defaultdict(set)
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            station = (row.get("Station") or "").strip()
            aliases_raw = (row.get("Aliases") or "").strip()
            if not station:
                continue
            station_norm = normalize_text(station)
            if not station_norm:
                continue
            for chunk in aliases_raw.split("|"):
                alias = normalize_text(chunk)
                if alias:
                    aliases_by_station[station_norm].add(alias)
    return aliases_by_station


def generate_station_aliases(station_name: str) -> Set[str]:
    """Generate conservative normalization aliases for punctuation/abbreviations."""
    def expand_phrase_variants(phrase_norm: str) -> Set[str]:
        """Expand one normalized phrase with conservative abbreviation variants."""
        out: Set[str] = set()
        tokens = phrase_norm.split()
        if not tokens:
            return out

        out.add(phrase_norm)
        repl_variants: List[List[str]] = [[]]
        for token in tokens:
            variants = {token}
            if token == "strasse":
                variants.update({"str", "str."})
            if token.endswith("strasse") and len(token) > len("strasse"):
                variants.update(
                    {token.replace("strasse", "str"), token.replace("strasse", "str.")}
                )
            if token == "platz":
                variants.update({"plz", "pl."})
            if token.endswith("platz") and len(token) > len("platz"):
                variants.add(token.replace("platz", "plz"))
            repl_variants.append(sorted(variants))

        prefixes = [""]
        for variants in repl_variants[1:]:
            next_prefixes: List[str] = []
            for prefix in prefixes:
                for candidate in variants:
                    next_prefixes.append(f"{prefix} {candidate}".strip())
            prefixes = next_prefixes[:40]

        for candidate in prefixes:
            alias_norm = normalize_text(candidate)
            if alias_norm:
                out.add(alias_norm)

        compact = "".join(tokens)
        if len(compact) >= 5:
            out.add(compact)
        return out

    aliases: Set[str] = set()
    normalized = normalize_text(station_name)
    if not normalized:
        return aliases

    base_phrases: Set[str] = {normalized}
    raw = station_name.strip()
    if "," in raw:
        right = raw.split(",", 1)[1].strip()
        right_norm = normalize_text(right)
        if right_norm:
            base_phrases.add(right_norm)

    for prefix in ("zurich", "zuerich", "winterthur"):
        if normalized.startswith(prefix + " "):
            short = normalized[len(prefix) + 1 :]
            if short:
                base_phrases.add(short)

    for phrase in base_phrases:
        aliases.update(expand_phrase_variants(phrase))

    return aliases


def sanitize_alias_conflicts(aliases_by_station: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    """Drop aliases mapping to multiple stations to prevent matcher ambiguity."""
    owners: Dict[str, Set[str]] = defaultdict(set)
    for station_norm, aliases in aliases_by_station.items():
        for alias in aliases:
            owners[alias].add(station_norm)

    conflicts = {alias for alias, stations in owners.items() if len(stations) > 1}
    if not conflicts:
        return aliases_by_station

    cleaned: Dict[str, Set[str]] = {}
    for station_norm, aliases in aliases_by_station.items():
        cleaned[station_norm] = {alias for alias in aliases if alias not in conflicts}
    return cleaned


def format_float(value: Optional[float]) -> str:
    """Format coordinates for CSV output."""
    if value is None:
        return ""
    text = f"{value:.7f}".rstrip("0").rstrip(".")
    return text


def build_station_table(
    ogd_stations: Dict[str, OGDStation],
    gtfs_stops: Dict[str, GTFSStop],
    stop_lines: Dict[str, Set[str]],
    stop_route_types: Dict[str, Counter[int]],
    existing_aliases: Dict[str, Set[str]],
) -> List[StationAggregate]:
    """Merge OGD base and GTFS enrichment into canonical station rows."""
    aggregates: Dict[str, StationAggregate] = {}

    # Seed with OGD stations (base catalog for Zurich scope).
    for station_norm, ogd in ogd_stations.items():
        aggregates[station_norm] = StationAggregate(
            station=ogd.name,
            normalized=station_norm,
            longitude=ogd.longitude,
            latitude=ogd.latitude,
            station_type=ogd.source_type,
            lines=set(ogd.source_lines),
            aliases=set(existing_aliases.get(station_norm, set())),
        )

    # Enrich/complete with GTFS stations and route metadata.
    for stop_id, stop in gtfs_stops.items():
        agg = aggregates.get(stop.normalized)
        if agg is None:
            agg = StationAggregate(
                station=stop.name,
                normalized=stop.normalized,
                longitude=None,
                latitude=None,
                aliases=set(existing_aliases.get(stop.normalized, set())),
            )
            aggregates[stop.normalized] = agg

        agg._gtfs_names[stop.name] += 1
        agg.add_coordinate_sample(stop.longitude, stop.latitude)
        agg.lines.update(stop_lines.get(stop_id, set()))
        agg._route_type_counts.update(stop_route_types.get(stop_id, Counter()))

    # Finalize fields and alias sets.
    existing_items = list(existing_aliases.items())
    for station_norm, agg in aggregates.items():
        agg.choose_final_name()
        agg.choose_final_coordinate()
        agg.choose_final_type()

        # Preserve existing aliases, add generated variants, and never include self alias.
        generated = generate_station_aliases(agg.station)
        agg.aliases.update(existing_aliases.get(station_norm, set()))
        # Carry forward aliases from older short station names that became locality-prefixed.
        for legacy_norm, legacy_aliases in existing_items:
            if legacy_norm == station_norm or station_norm.endswith(f" {legacy_norm}"):
                agg.aliases.update(legacy_aliases)
                agg.aliases.add(legacy_norm)
        agg.aliases.update(generated)
        agg.aliases.discard(station_norm)

    alias_map = {station_norm: set(agg.aliases) for station_norm, agg in aggregates.items()}
    alias_map = sanitize_alias_conflicts(alias_map)
    for station_norm, agg in aggregates.items():
        agg.aliases = alias_map.get(station_norm, set())

    # Deterministic output order.
    rows = list(aggregates.values())
    rows.sort(key=lambda row: (row.station.lower(), row.normalized))
    return rows


def write_station_csv(path: Path, rows: List[StationAggregate]) -> None:
    """Write final station CSV in the exact schema expected downstream."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            lines = sorted(
                {line.strip() for line in row.lines if line and line.strip()},
                key=natural_sort_key,
            )
            aliases = sorted(
                {alias.strip() for alias in row.aliases if alias and alias.strip()},
                key=natural_sort_key,
            )
            writer.writerow(
                {
                    "Station": row.station,
                    "Type": row.station_type,
                    "line": "|".join(lines),
                    "Longitude": format_float(row.longitude),
                    "Latitude": format_float(row.latitude),
                    "Aliases": "|".join(aliases),
                }
            )


def parse_args() -> argparse.Namespace:
    """CLI parameters for manual refresh runs."""
    parser = argparse.ArgumentParser(
        description="Rebuild Zurich station reference CSV from official online open-data sources."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/zurich_stations.csv"),
        help="Output CSV path (default: data/zurich_stations.csv).",
    )
    parser.add_argument(
        "--preserve-aliases-from",
        type=Path,
        default=Path("data/zurich_stations.csv"),
        help="Existing CSV used to preserve/merge manual aliases.",
    )
    parser.add_argument(
        "--city-api-base",
        default=DEFAULT_CITY_API_BASE,
        help="CKAN API base for Zurich OGD package.",
    )
    parser.add_argument(
        "--city-package",
        default=DEFAULT_CITY_PACKAGE,
        help="Zurich OGD package id containing public transport stops.",
    )
    parser.add_argument(
        "--gtfs-api-base",
        default=DEFAULT_GTFS_API_BASE,
        help="CKAN API base for GTFS package.",
    )
    parser.add_argument(
        "--gtfs-package",
        default=DEFAULT_GTFS_PACKAGE,
        help="GTFS package id (switch yearly as needed).",
    )
    parser.add_argument("--min-lat", type=float, default=DEFAULT_MIN_LAT)
    parser.add_argument("--max-lat", type=float, default=DEFAULT_MAX_LAT)
    parser.add_argument("--min-lon", type=float, default=DEFAULT_MIN_LON)
    parser.add_argument("--max-lon", type=float, default=DEFAULT_MAX_LON)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and print summary without writing output file.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for manual online refresh of station reference data."""
    args = parse_args()

    if args.min_lat >= args.max_lat or args.min_lon >= args.max_lon:
        raise ValueError("Invalid bounding box: min values must be lower than max values.")

    existing_aliases = load_existing_aliases(args.preserve_aliases_from)

    # Resolve CKAN resource URLs dynamically to support yearly/package updates.
    city_meta, city_api_used, city_package_used = fetch_package_with_fallback(
        [args.city_api_base, "https://ckan.opendata.swiss/api/3/action"],
        args.city_package,
    )
    city_resources = city_meta.get("result", {}).get("resources", [])
    ogd_stations, city_resource_url = load_ogd_stations_from_resources(city_resources)

    gtfs_meta, gtfs_api_used, gtfs_package_used = fetch_package_with_fallback(
        [
            args.gtfs_api_base,
            "https://ckan.opendata.swiss/api/3/action",
            "https://opendata.swiss/api/3/action",
            "https://data.opentransportdata.swiss/api/3/action",
        ],
        args.gtfs_package,
    )
    gtfs_resources = gtfs_meta.get("result", {}).get("resources", [])
    gtfs_zip_url = choose_ckan_resource(gtfs_resources, kind="gtfs_zip")

    gtfs_stops, stop_lines, stop_route_types = load_gtfs_stops_and_routes(
        gtfs_zip=fetch_bytes(gtfs_zip_url),
        min_lat=args.min_lat,
        max_lat=args.max_lat,
        min_lon=args.min_lon,
        max_lon=args.max_lon,
        ogd_names=set(ogd_stations.keys()),
    )

    rows = build_station_table(
        ogd_stations=ogd_stations,
        gtfs_stops=gtfs_stops,
        stop_lines=stop_lines,
        stop_route_types=stop_route_types,
        existing_aliases=existing_aliases,
    )

    with_coords = sum(1 for row in rows if row.longitude is not None and row.latitude is not None)
    with_lines = sum(1 for row in rows if row.lines)
    with_aliases = sum(1 for row in rows if row.aliases)

    print(f"City package: {city_package_used} (API: {city_api_used})")
    print(f"GTFS package: {gtfs_package_used} (API: {gtfs_api_used})")
    print(f"City resource URL: {city_resource_url}")
    print(f"GTFS resource URL: {gtfs_zip_url}")
    print(f"Stations built: {len(rows)}")
    print(f"Stations with coordinates: {with_coords}")
    print(f"Stations with lines: {with_lines}")
    print(f"Stations with aliases: {with_aliases}")

    if args.dry_run:
        print("Dry run enabled. No file written.")
        return

    write_station_csv(args.output, rows)
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
