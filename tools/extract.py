#!/usr/bin/env python3
"""
Extract JSON payloads from hex-dump WebSocket log files.

Input format (example):
    2026-07-01T23:03:44.873Z  INFO  T_2 WebSocket >>> DEVICE_TO_BACKEND BINARY payload for connection AA:BB:CC:33:44:CC-1782947015115 (3733 bytes):
    2026-07-01T23:03:44.873Z  INFO  T_2 000| 7B 22 66 72 6F 6D 22 3A 20 22 75 62 6E 74 5F 61  {"from": "ubnt_a
    ...

Output: one line per JSON record, prefixed with the connection/direction details,
in the form:
    <timestamp>\t<direction>\t<connection>\t<size_bytes>\t<json>

Usage:
    python3 extract_ws_json.py input.log > output.jsonl
    python3 extract_ws_json.py input.log --json-only > output.jsonl   # just the JSON, tab-prefixed with connection
"""

import re
import sys
import argparse

# Matches the header line that starts a hex-dump block, e.g.:
# 2026-07-01T23:03:44.873Z  INFO  T_2 WebSocket >>> DEVICE_TO_BACKEND BINARY payload for connection AA:BB:CC:33:44:CC-1782947015115 (3733 bytes):
HEADER_RE = re.compile(
    r'^(?P<ts>\S+)\s+INFO\s+(?P<thread>\S+)\s+WebSocket\s+(?:(?P<arrow>>>>|<<<)\s+)?'
    r'(?P<direction>\S+)\s+BINARY payload for connection\s+(?P<connection>\S+)\s+'
    r'\((?P<size>\d+) bytes\):\s*$'
)

# Matches a hex-dump data line, e.g.:
# 2026-07-01T23:03:44.873Z  INFO  T_2 000| 7B 22 66 72 6F 6D 22 3A 20 22 75 62 6E 74 5F 61  {"from": "ubnt_a
DATA_RE = re.compile(
    r'^\S+\s+INFO\s+\S+\s+(?P<offset>[0-9A-Fa-f]+)\|\s(?P<rest>.*)$'
)

HEX_TOKEN_RE = re.compile(r'^[0-9A-Fa-f]{2}$')


def extract_hex_bytes(rest):
    """Given the text after 'OFFSET| ', pull out the leading run of 2-char
    hex tokens (up to 16) and return the decoded bytes."""
    tokens = rest.split(' ')
    hex_tokens = []
    for tok in tokens:
        if tok == '':
            continue
        if HEX_TOKEN_RE.match(tok) and len(hex_tokens) < 16:
            hex_tokens.append(tok)
        else:
            break
    if not hex_tokens:
        return b''
    return bytes.fromhex(''.join(hex_tokens))


def parse_log(lines):
    """Yield dicts with header info + decoded json text for each block."""
    current_header = None
    current_bytes = bytearray()

    def flush():
        if current_header is not None:
            try:
                text = current_bytes.decode('utf-8', errors='replace')
            except Exception:
                text = current_bytes.decode('latin1', errors='replace')
            yield_record = dict(current_header)
            yield_record['json_text'] = text
            return yield_record
        return None

    records = []
    for line in lines:
        line = line.rstrip('\n')
        if not line.strip():
            continue

        h = HEADER_RE.match(line)
        if h:
            # flush previous block
            if current_header is not None:
                rec = flush()
                if rec:
                    records.append(rec)
            current_header = h.groupdict()
            current_bytes = bytearray()
            continue

        d = DATA_RE.match(line)
        if d and current_header is not None:
            current_bytes.extend(extract_hex_bytes(d.group('rest')))
            continue

        # Any other line type (unrecognized) - ignore, but flush current block
        # since it signals the block segment ended (defensive).
        # We don't flush here to allow blank/other benign lines within blocks.

    if current_header is not None:
        rec = flush()
        if rec:
            records.append(rec)

    return records


def normalize_json(text):
    """Try to compact/validate the JSON; fall back to raw text if it fails."""
    import json
    try:
        obj = json.loads(text)
        return json.dumps(obj, separators=(',', ':'))
    except Exception:
        # Trim trailing null bytes / whitespace and retry once
        stripped = text.strip('\x00').strip()
        try:
            obj = json.loads(stripped)
            return json.dumps(obj, separators=(',', ':'))
        except Exception:
            return text  # give up, return as-is (still one line, newlines stripped)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('input', help='Path to the hex-dump log file (use - for stdin)')
    ap.add_argument('--json-only', action='store_true',
                     help='Only output <connection details>\\t<json> instead of full metadata columns')
    ap.add_argument('--raw-json', action='store_true',
                     help='Do not attempt to re-serialize/compact JSON (keep original text verbatim, minus newlines)')
    args = ap.parse_args()

    if args.input == '-':
        lines = sys.stdin.readlines()
    else:
        with open(args.input, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()

    records = parse_log(lines)

    for rec in records:
        json_text = rec['json_text']
        # Collapse to a single line no matter what
        single_line = json_text.replace('\r', '').replace('\n', '\\n')
        if not args.raw_json:
            single_line = normalize_json(single_line.replace('\\n', '\n')).replace('\n', '\\n')

        if args.json_only:
            prefix = f"{rec['direction']} {rec['connection']}"
            print(f"{prefix}\t{single_line}")
        else:
            print(f"{rec['ts']}\t{rec['direction']}\t{rec['connection']}\t{rec['size']}\t{single_line}")


if __name__ == '__main__':
    main()

