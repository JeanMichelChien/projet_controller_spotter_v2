# Zurich Telegram Station Mapper

This project extracts station-level events from Telegram message history and visualizes them on a Streamlit map.

Architecture diagram: [ARCHITECTURE.md](ARCHITECTURE.md)

## Data flow

1. Local raw messages (ignored): `exports/messages.jsonl`
2. Fixed station reference: `data/zurich_stations.csv` (includes optional `Aliases` column, pipe-separated)
3. Derived repo-visible table: `data/station_events.csv`
4. Streamlit app reads only `data/station_events.csv`

Matching order in the build step is:
`exact` -> `alias` -> `event_scope classifier` (transport/non-transport) -> conservative `fuzzy` -> `unmatched`.

Messages classified as non-transport are kept as unmatched with `match_method=non_transport`.

## Build extracted datasource

```bash
.venv/bin/python scripts/build_station_events.py \
  --input exports/messages.jsonl \
  --stations data/zurich_stations.csv \
  --output data/station_events.csv
```

The build also writes `exports/unmatched_messages.csv` (unless disabled with empty `--unmatched-output`)
including `event_scope` and `match_method` columns for review.

## Alias curation loop (manual approval required)

1. Generate suggestions from unmatched messages:

```bash
.venv/bin/python scripts/suggest_station_aliases.py suggest \
  --unmatched exports/unmatched_messages.csv \
  --stations data/zurich_stations.csv \
  --output exports/alias_suggestions_review.csv
```

2. Review `exports/alias_suggestions_review.csv` and set `review_status=approve` for accepted rows.

3. Apply only approved aliases to the stations file:

```bash
.venv/bin/python scripts/suggest_station_aliases.py apply \
  --stations data/zurich_stations.csv \
  --review exports/alias_suggestions_review.csv
```

## Rebuild station reference from online open data

This manual refresh command rebuilds `data/zurich_stations.csv` from official Zurich OGD stops
plus Swiss GTFS route metadata while preserving aliases in the same CSV.

```bash
.venv/bin/python scripts/rebuild_station_reference_online.py \
  --output data/zurich_stations.csv \
  --preserve-aliases-from data/zurich_stations.csv \
  --gtfs-package timetable-2026-gtfs2020
```

Optional:
- `--dry-run` to print counts without writing.
- `--min-lat/--max-lat/--min-lon/--max-lon` to override geographic scope.
- `--gtfs-package` to switch yearly package ids (`timetable-*` and `fahrplan-*` variants are both supported).

## Run app

```bash
.venv/bin/streamlit run streamlit_app.py
```

## Data quality and statistical bias

**Important:** this dataset reflects reported messages, not true control incidence.

- Exposure bias: more passengers in transport areas can produce more reports, so higher counts may reflect traffic volume rather than higher control intensity.
- Reporting bias: places with more active group members are overrepresented; quiet zones may be underreported.
- Time bias: reporting behavior changes by hour/day and by user availability.
- Selection bias: only messages visible to the exporting account are included (missing/deleted/inaccessible posts are not present).
- Extraction bias: station matching (exact/alias/fuzzy) can create false positives and false negatives.

Use charts and map outputs as relative signal indicators, not as ground-truth operational statistics.

## Disclaimer shown in app

These randomized data are presented for educational purposes only and are not representative of reality. The author of these visualisations is not responsible for the use of this information by third parties.
