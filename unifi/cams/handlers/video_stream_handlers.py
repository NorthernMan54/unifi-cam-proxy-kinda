import json
import os
import signal
import subprocess
import sys
from typing import Any


class VideoStreamHandlers:
    """Mixin class providing video stream management functionality"""

    def get_extra_ffmpeg_args(self, stream_index: str = "") -> str:
        return self.args.ffmpeg_args

    def get_base_ffmpeg_args(self, stream_index: str = "") -> str:
        if self.args.ffmpeg_base_args is not None:
            return self.args.ffmpeg_base_args

        base_args = [
            "-avoid_negative_ts",
            "make_zero",
            "-fflags",
            "+genpts+discardcorrupt",
            "-use_wallclock_as_timestamps 1",
        ]

        try:
            output = subprocess.check_output(["ffmpeg", "-h", "full"])
            if b"stimeout" in output:
                base_args.append("-stimeout 15000000")
            else:
                base_args.append("-timeout 15000000")
        except subprocess.CalledProcessError:
            self.logger.exception("Could not check for ffmpeg options")

        return " ".join(base_args)

    def probe_video_resolution(self, stream_index: str, source_url: str) -> tuple[int, int]:
        """Probe video source to detect width and height using ffprobe"""
        # Get default resolution for this stream
        default_width, default_height = self._detected_resolutions[stream_index]
        
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height',
                '-of', 'json',
                '-rtsp_transport', self.args.rtsp_transport,
                source_url
            ]
            self.logger.info(f"Probing {stream_index} source: {source_url}")
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=15
            )
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                if data.get('streams') and len(data['streams']) > 0:
                    width = data['streams'][0].get('width', default_width)
                    height = data['streams'][0].get('height', default_height)
                    self.logger.info(f"Detected {stream_index} resolution: {width}x{height}")
                    return width, height
                    
        except subprocess.TimeoutExpired:
            self.logger.warning(f"{stream_index} probe timed out after 15 seconds, using defaults")
        except json.JSONDecodeError as e:
            self.logger.warning(f"Could not parse ffprobe output for {stream_index}: {e}, using defaults")
        except Exception as e:
            self.logger.warning(f"Could not probe {stream_index} source: {e}, using defaults")
        
        # Fallback to defaults for this stream
        self.logger.info(f"Using default resolution for {stream_index}: {default_width}x{default_height}")
        return default_width, default_height

    async def start_video_stream(
        self, stream_index: str, stream_name: str, destination: tuple[str, int]
    ):
        has_spawned = stream_index in self._ffmpeg_handles
        is_dead = has_spawned and self._ffmpeg_handles[stream_index].poll() is not None

        if not has_spawned or is_dead:
            source = await self.get_stream_source(stream_index)
            cmd = (
                f"AV_LOG_FORCE_NOCOLOR=1 ffmpeg -nostdin -loglevel level+{self.args.loglevel} -y"
                f" {self.get_base_ffmpeg_args(stream_index)} -rtsp_transport"
                f' {self.args.rtsp_transport} -i "{source}"'
                f" {self.get_extra_ffmpeg_args(stream_index)} -metadata"
                f" streamName={stream_name} -f {self.args.format} - "
                f" | {sys.executable} -m unifi.clock_sync --timestamp-modifier {self.args.timestamp_modifier}"
                f" | nc"
                f" {destination[0]} {destination[1]}"
            )

            if is_dead:
                exit_code = self._ffmpeg_handles[stream_index].poll()
                self.logger.warning(f"Previous ffmpeg process for {stream_index} died with exit code {exit_code}.")

            self.logger.info(
                f"Spawning ffmpeg for {stream_index} ({stream_name}): {cmd}"
            )
            # Start process in a new process group so we can kill the entire pipeline
            self._ffmpeg_handles[stream_index] = subprocess.Popen(
                cmd, 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL, 
                shell=True,
                preexec_fn=os.setsid  # Create new process group
            )

    def stop_video_stream(self, stream_index: str):
        if stream_index in self._ffmpeg_handles:
            self.logger.info(f"Stopping stream {stream_index}")
            proc = self._ffmpeg_handles[stream_index]
            
            # Check if process is already dead
            if proc.poll() is not None:
                self.logger.debug(f"Process for {stream_index} already terminated with code {proc.poll()}")
                del self._ffmpeg_handles[stream_index]
                return
            
            try:
                # Terminate the process group to kill all processes in the pipeline
                pgid = os.getpgid(proc.pid)
                self.logger.debug(f"Sending SIGTERM to process group {pgid} for {stream_index}")
                os.killpg(pgid, signal.SIGTERM)
                
                # Wait for graceful shutdown
                try:
                    proc.wait(timeout=2)
                    self.logger.debug(f"Stream {stream_index} terminated gracefully")
                except subprocess.TimeoutExpired:
                    self.logger.warning(f"Stream {stream_index} did not terminate gracefully, sending SIGKILL")
                    try:
                        os.killpg(pgid, signal.SIGKILL)
                        proc.wait(timeout=1)
                    except (ProcessLookupError, subprocess.TimeoutExpired):
                        pass
                        
            except (ProcessLookupError, PermissionError, AttributeError, OSError) as e:
                self.logger.debug(f"Error stopping {stream_index}: {e}, trying proc.kill()")
                # Fall back to killing just the parent process
                try:
                    proc.kill()
                    proc.wait(timeout=1)
                except Exception:
                    pass
            
            # Remove from handles
            del self._ffmpeg_handles[stream_index]

    def close_streams(self):
        for stream in self._ffmpeg_handles:
            self.stop_video_stream(stream)
