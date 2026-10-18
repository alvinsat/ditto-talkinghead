import os
import subprocess


class VideoWriterByImageIO:
    def __init__(self, video_path, fps=25, **kwargs):
        codec = kwargs.get("vcodec", "libx264")  # default is libx264 encoding
        self.pixelformat = kwargs.get("pixelformat", "yuv420p")
        self.crf = str(kwargs.get("crf", 18))
        self.video_path = video_path
        self.fps = str(fps)
        self.codec = codec
        self.process = None

        output_directory = os.path.dirname(video_path)
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)

    def __call__(self, img, fmt="bgr"):
        if fmt == "bgr":
            frame = img[..., ::-1]
        else:
            frame = img
        if self.process is None:
            height, width = frame.shape[:2]
            self.process = subprocess.Popen(
                [
                    "ffmpeg",
                    "-loglevel", "error",
                    "-y",
                    "-f", "rawvideo",
                    "-vcodec", "rawvideo",
                    "-pix_fmt", "rgb24",
                    "-s", f"{width}x{height}",
                    "-r", self.fps,
                    "-i", "-",
                    "-an",
                    "-c:v", self.codec,
                    "-pix_fmt", self.pixelformat,
                    "-crf", self.crf,
                    self.video_path,
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        try:
            self.process.stdin.write(frame.tobytes())
        except (BrokenPipeError, OSError) as error:
            raise RuntimeError(f"FFmpeg stopped while writing {self.video_path}") from error

    def close(self):
        if self.process is None:
            return

        self.process.stdin.close()
        stderr = self.process.stderr.read()
        return_code = self.process.wait()
        if return_code:
            error = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"FFmpeg failed to write {self.video_path} "
                f"(exit code {return_code}): {error}"
            )
