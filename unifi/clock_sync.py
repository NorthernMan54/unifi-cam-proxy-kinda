"""
Helper program to inject absolute wall clock time into FLV stream for recordings
"""

import argparse
import struct
import sys
import time

from flvlib3.astypes import FLVObject
from flvlib3.primitives import make_ui8, make_ui32
from flvlib3.tags import create_script_tag

# FIX: the two byte sequences below aren't arbitrary padding -- read as
# big-endian 24-bit integers they declare the CLOCK RATE the following
# timestamp is expressed in: [1,95,144] = 90000, [0,43,17] = 11025.
# Every call site previously computed the actual tick value as
# `elapsed_seconds * args.timestamp_modifier * 1000` regardless of which
# branch fired -- i.e. always on a 90000Hz basis (timestamp_modifier
# defaults to 90, so *1000 = 90000). That's correct for video (is_packet=True,
# rate 90000) but wrong for everything else (audio, script/metadata tags,
# the injected onClockSync/onMpma sync tags), which all declare an 11025Hz
# rate yet received a tick count computed on the 90000Hz basis. Downstream,
# reconstructing wall time as ticks/declared_rate then runs audio at
# 90000/11025 =~ 8.163x real speed -- matching the observed FeedData
# audio/video desync growing at ~8x real-time exactly.
#
# Fix: compute the tick value INSIDE this function, using the rate that
# matches whichever branch actually fires, so it's no longer possible for a
# caller to pass a mismatched pre-scaled value.
VIDEO_CLOCK_RATE = 90000
AUDIO_CLOCK_RATE = 11025


def read_bytes(source, num_bytes):
    read_bytes = 0
    buf = b""
    while read_bytes < num_bytes:
        d_in = source.read(num_bytes - read_bytes)
        if d_in:
            read_bytes += len(d_in)
            buf += d_in
        else:
            return buf
    return buf


def write(data):
    sys.stdout.buffer.write(data)


def write_log(data):
    sys.stderr.buffer.write(f"{data}\n".encode())


def write_timestamp_trailer(is_packet, elapsed_seconds):
    """
    Write the 15-byte clock-sync trailer.

    Args:
        is_packet: True for video tags (FLV packet_type == 9), False for
            audio/script/metadata tags.
        elapsed_seconds: real elapsed time (now - start), UNSCALED. The
            correct clock rate is applied internally based on is_packet,
            rather than requiring the caller to pre-multiply by the right
            modifier (which was the source of the audio/video desync bug --
            see module-level comment).
    """
    write(make_ui8(0))
    if is_packet:
        write(bytes([1, 95, 144, 0, 0, 0, 0, 0, 0, 0, 0]))  # declares 90000Hz
        rate = VIDEO_CLOCK_RATE
    else:
        write(bytes([0, 43, 17, 0, 0, 0, 0, 0, 0, 0, 0]))  # declares 11025Hz
        rate = AUDIO_CLOCK_RATE

    write(make_ui32(int(elapsed_seconds * rate)))


def main(args):
    source = sys.stdin.buffer

    header = read_bytes(source, 3)

    if header != b"FLV":
        print("Not a valid FLV file")
        return
    write(header)

    # Skip rest of FLV header
    write(read_bytes(source, 1))
    read_bytes(source, 1)
    # Write custom bitmask for FLV type
    write(make_ui8(7))
    write(read_bytes(source, 4))

    # Tag 0 previous size
    write(read_bytes(source, 4))

    last_ts = time.time()
    start = time.time()
    i = 0
    while True:
        # Packet structure from Wikipedia:
        #
        # Size of previous packet	uint32_be	0	For first packet set to NULL
        #
        # Packet Type	uint8	18	For first packet set to AMF Metadata
        # Payload Size	uint24_be	varies	Size of packet data only
        # Timestamp Lower	uint24_be	0	For first packet set to NULL
        # Timestamp Upper	uint8	0	Extension to create a uint32_be value
        # Stream ID	uint24_be	0	For first stream of same type set to NULL
        #
        # Payload Data	freeform	varies	Data as defined by packet type

        header = read_bytes(source, 12)
        if len(header) != 12:
            write(header)
            return

        # Packet type
        packet_type = header[0]

        # Get payload size to know how many bytes to read
        high, low = struct.unpack(">BH", header[1:4])
        payload_size = (high << 16) + low

        # Get timestamp to inject into clock sync tag
        low_high = header[4:8]
        combined = bytes([low_high[3]]) + low_high[:3]
        timestamp = struct.unpack(">i", combined)[0]

        now = time.time()
        elapsed = now - start
        if not last_ts or now - last_ts >= 5:
            last_ts = now
            # Insert a custom packet every so often for time synchronization
            data = FLVObject()
            data["streamClock"] = int(timestamp)
            data["streamClockBase"] = 0
            data["wallClock"] = now * 1000
            packet_to_inject = create_script_tag("onClockSync", data, timestamp)
            write(packet_to_inject)

            # Write 15 byte trailer (script/metadata tag -> audio-rate branch)
            write_timestamp_trailer(False, elapsed)

            # Write mpma tag
            # {'cs': {'cur': 1500000.0,
            #         'max': 1500000.0,
            #         'min': 32000.0},
            #  'm': {'cur': 750000.0,
            #        'max': 1500000.0,
            #        'min': 750000.0},
            #  'r': 0.0,
            #  'sp': {'cur': 1500000.0,
            #         'max': 1500000.0,
            #         'min': 150000.0},
            #  't': 750000.0}

            data = FLVObject()
            data["cs"] = FLVObject()
            data["cs"]["cur"] = 1500000
            data["cs"]["max"] = 1500000
            data["cs"]["min"] = 1500000

            data["m"] = FLVObject()
            data["m"]["cur"] = 1500000
            data["m"]["max"] = 1500000
            data["m"]["min"] = 1500000
            data["r"] = 0

            data["sp"] = FLVObject()
            data["sp"]["cur"] = 1500000
            data["sp"]["max"] = 1500000
            data["sp"]["min"] = 1500000
            data["t"] = 75000.0
            packet_to_inject = create_script_tag("onMpma", data, 0)

            write(packet_to_inject)

            # Write 15 byte trailer (script/metadata tag -> audio-rate branch)
            write_timestamp_trailer(False, elapsed)

            # Write rest of original packet minus previous packet size
            write(header)
            write(read_bytes(source, payload_size))
        else:
            # Write the original packet
            write(header)
            write(read_bytes(source, payload_size))

        # Write previous packet size
        write(read_bytes(source, 3))

        # Write 15 byte trailer -- is_packet determines both the declared
        # rate marker AND now the actual scale applied (see fix above):
        # video tags (packet_type == 9) get the 90000Hz-scaled value,
        # everything else (audio, script) gets the 11025Hz-scaled value.
        write_timestamp_trailer(packet_type == 9, elapsed)

        # Write mpma tag
        i += 1


def parse_args():
    parser = argparse.ArgumentParser(description="Modify Protect FLV stream")
    parser.add_argument(
        "--timestamp-modifier",
        type=int,
        default="90",
        help=(
            "DEPRECATED / no longer used to scale the timestamp trailer -- "
            "kept only for CLI compatibility with existing callers. The "
            "video clock rate (90000Hz) and audio clock rate (11025Hz) are "
            "now both fixed constants applied internally based on tag type, "
            "since using a single shared modifier for both was the cause of "
            "the audio/video clock desync bug (see write_timestamp_trailer)."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())