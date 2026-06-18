# data/

Default location for large local artifacts that are NOT committed:

- `action100m_index.db` — Action100M SQLite index (built locally; see
  docs/THIRD_PARTY.md). Override the path with the `ACTION100M_DB` env var.

This directory is intentionally near-empty in the repo. The pipeline and
filter tools create files here on demand.
