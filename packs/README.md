# Roof pack checksums

One small JSON file per roof pack: file name, size in bytes, SHA-256 checksum, number of roofs and roofs per
source. The zips themselves are on release `data-v1` of this repository (too large for git). The notebooks
download a zip, compare its SHA-256 with the value here and refuse a file that does not match.
