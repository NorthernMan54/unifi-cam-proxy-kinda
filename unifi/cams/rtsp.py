import argparse
import logging
import subprocess
import tempfile
from pathlib import Path

from aiohttp import web

from unifi.cams.base import UnifiCamBase


class RTSPCam(UnifiCamBase):
    def __init__(self, args: argparse.Namespace, logger: logging.Logger) -> None:
        super().__init__(args, logger)
        self.args = args
        self.event_id = 0
        self.snapshot_dir = tempfile.mkdtemp()
        self.snapshot_stream = None
        self.runner = None
        self.stream_source = dict()
        self.configured_streams = set()  # Track which streams were explicitly configured
        
        # Support both old --source format and new --video1/2/3 format
        if hasattr(args, 'source') and args.source:
            # Legacy format: --source URL1 [URL2] [URL3]
            # In legacy mode, all three streams are considered configured
            for i, stream_index in enumerate(["video1", "video2", "video3"]):
                if i < len(args.source):
                    self.stream_source[stream_index] = args.source[i]
                else:
                    self.stream_source[stream_index] = args.source[0]
                self.configured_streams.add(stream_index)
        else:
            # New format: --video1 URL1 [--video2 URL2] [--video3 URL3]
            if not args.video1:
                raise ValueError("Either --source or --video1 must be provided")
            
            # Only set stream sources for explicitly configured streams
            self.stream_source["video1"] = args.video1
            self.configured_streams.add("video1")
            
            if args.video2:
                self.stream_source["video2"] = args.video2
                self.configured_streams.add("video2")
            if args.video3:
                self.stream_source["video3"] = args.video3
                self.configured_streams.add("video3")
        
        if not self.args.snapshot_url:
            self.start_snapshot_stream()

    @classmethod
    def add_parser(cls, parser: argparse.ArgumentParser) -> None:
        super().add_parser(parser)
        parser.add_argument(
            "--source",
            "-s",
            nargs="+",
            required=False,
            help="Source(s) for up to three streams in order of descending quality (deprecated, use --video1/2/3)",
        )
        parser.add_argument(
            "--video1",
            type=str,
            required=False,
            help="RTSP source for high quality video stream (required if --source not provided)",
        )
        parser.add_argument(
            "--video2",
            type=str,
            required=False,
            help="RTSP source for medium quality video stream (optional, defaults to video1)",
        )
        parser.add_argument(
            "--video3",
            type=str,
            required=False,
            help="RTSP source for low quality video stream (optional, defaults to video1)",
        )
        parser.add_argument(
            "--http-api",
            default=0,
            type=int,
            help="Specify a port number to enable the HTTP API (default: disabled)",
        )
        parser.add_argument(
            "--snapshot-url",
            "-i",
            default=None,
            type=str,
            required=False,
            help="HTTP endpoint to fetch snapshot image from",
        )

    def start_snapshot_stream(self) -> None:
        if not self.snapshot_stream or self.snapshot_stream.poll() is not None:
            # Use video3 (lowest quality) for snapshots
            snapshot_source = self.stream_source.get("video3", self.stream_source["video1"])
            cmd = (
                f"AV_LOG_FORCE_NOCOLOR=1 ffmpeg -loglevel level+{self.args.loglevel} "
                f"-nostdin -y -re -rtsp_transport {self.args.rtsp_transport} "
                f'-i "{snapshot_source}" '
                "-r 1 "
                f"-update 1 {self.snapshot_dir}/screen.jpg"
            )
            self.logger.info(f"Spawning stream for snapshots: {cmd}")
            self.snapshot_stream = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True
            )

    async def get_snapshot(self) -> Path:
        img_file = Path(self.snapshot_dir, "screen.jpg")
        if self.args.snapshot_url:
            await self.fetch_to_file(self.args.snapshot_url, img_file)
        else:
            self.start_snapshot_stream()
        return img_file

    async def run(self) -> None:
        if self.args.http_api:
            self.logger.info(f"Enabling HTTP API on port {self.args.http_api}")

            app = web.Application()

            async def start_motion(request):
                self.logger.debug("Starting motion")
                await self.trigger_motion_start()
                return web.Response(text="ok")

            async def stop_motion(request):
                self.logger.debug("Starting motion")
                await self.trigger_motion_stop()
                return web.Response(text="ok")

            app.add_routes([web.get("/start_motion", start_motion)])
            app.add_routes([web.get("/stop_motion", stop_motion)])

            self.runner = web.AppRunner(app)
            await self.runner.setup()
            site = web.TCPSite(self.runner, port=self.args.http_api)
            await site.start()

    async def close(self) -> None:
        await super().close()
        if self.runner:
            await self.runner.cleanup()

        if self.snapshot_stream:
            self.snapshot_stream.kill()

    async def get_stream_source(self, stream_index: str) -> str:
        # Return source if stream is explicitly configured
        if stream_index in self.stream_source:
            return self.stream_source[stream_index]
        # For unconfigured streams, return video1 as fallback for actual streaming
        # This allows video2/video3 to use video1's stream if they weren't configured
        return self.stream_source.get("video1", "")
