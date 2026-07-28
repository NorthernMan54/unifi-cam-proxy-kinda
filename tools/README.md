# Tools Directory

This directory contains utility scripts and tools used for data extraction, processing, and analysis across the project.

## Structure

```
tools/
├── README.md              # This file
└── extract.py             # WebSocket hex-dump JSON extractor
```

## Tools Overview

| Tool | Description |
|------|-------------|
| `extract.py` | Extracts JSON payloads from hex-dump WebSocket log files |

---

## extract.py - WebSocket Hex-Dump JSON Extractor

### Overview

`extract.py` parses WebSocket log files containing hex-dump formatted binary payloads and extracts JSON data from them. The tool reads log files with hex-encoded WebSocket messages and outputs clean JSON records.

### Create Input log file

This is from my UCG

1 - Edit /usr/share/ds/ds.json and change log_websocket_payload to true
2 - Restart the `ds` process

```
ps -ef | grep ds
```
Then kill the ds process and not the monitor.

### Input Format

The tool expects log files in the following format:

```
2026-07-01T23:03:44.873Z  INFO  T_2 WebSocket >>> DEVICE_TO_BACKEND BINARY payload for connection AA:BB:CC:33:44:CC-1782947015115 (3733 bytes):
2026-07-01T23:03:44.873Z  INFO  T_2 000| 7B 22 66 72 6F 6D 22 3A 20 22 75 62 6E 74 5F 61  {"from": "ubnt_a
2026-07-01T23:03:44.873Z  INFO  T_2 010| 74 65 64 22 3A 20 74 72 75 65 2C 20 22 74 6F 22 3A 20 22 31 32 33 22
```

### Output Format

**Default (full metadata):**
```
<timestamp>\t<direction>\t<connection>\t<size_bytes>\t<json>
```

**JSON-only mode:**
```
<direction> <connection>\t<json>
```

### Usage

```bash
# Extract JSON with full metadata
python3 tools/extract.py input.log > output.jsonl

# Extract JSON only (connection + JSON)
python3 tools/extract.py input.log --json-only > output.jsonl

# Read from stdin
cat input.log | python3 tools/extract.py - > output.jsonl

# Keep original JSON text without normalization
python3 tools/extract.py input.log --raw-json
```

### Options

| Option | Description |
|--------|-------------|
| `input` | Path to hex-dump log file (use `-` for stdin) |
| `--json-only` | Output only `<direction> <connection>\t<json>` |
| `--raw-json` | Don't re-serialize/compact JSON (keep original text) |

### Example

```bash
# Basic extraction
python3 tools/extract.py /app/unifi/logs/ws_dump.log > extracted.jsonl

# Extract just the JSON payloads with connection info
python3 tools/extract.py /app/unifi/logs/ws_dump.log --json-only > json_payloads.jsonl

# Process from stdin
cat /app/unifi/logs/ws_dump.log | python3 tools/extract.py -
```

### Implementation Details

- **Hex decoding**: Extracts up to 16 consecutive 2-character hex tokens per data line
- **Encoding**: Attempts UTF-8, falls back to Latin-1
- **JSON normalization**: Compacts JSON output using `separators=(',', ':')`
- **Error handling**: Handles invalid JSON gracefully, returns raw text if parsing fails
- **Line collapsing**: Converts newlines within JSON to escaped `\n`

### File Location

```
tools/extract.py
```

---

## Contributing

To add new tools to this directory:

1. Create a Python script with a shebang line
2. Add documentation in this README
3. Ensure tools are self-contained and have clear usage instructions

---

## License

[Insert license information here]