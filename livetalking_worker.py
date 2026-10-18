import argparse
import queue
import socket
import struct
import threading
import time
import os

import numpy as np
import torch

from stream_pipeline_online import StreamSDK


AUDIO_PACKET_SAMPLES = 320
AUDIO_PACKET_BYTES = AUDIO_PACKET_SAMPLES * 2
DITTO_FRAME_SAMPLES = 640
DITTO_STEP_SAMPLES = 5 * DITTO_FRAME_SAMPLES
DITTO_PREFIX_SAMPLES = 2000
DITTO_WINDOW_SAMPLES = 6480
SPEECH_RMS_THRESHOLD = 0.003
SPEECH_HANGOVER_SECONDS = 0.4


class DittoLiveWorker:
    def __init__(self, args):
        self.frames = queue.Queue(maxsize=8)
        self.audio_packets = queue.Queue(maxsize=25)
        self.motion_audio = queue.Queue(maxsize=15)
        self.worker_error = None
        self.stop_event = threading.Event()
        self.sdk = StreamSDK(
            args.cfg_pkl,
            args.data_root,
            frame_sink=self._publish_frame,
        )
        self.sdk.setup(args.source_path, output_path=None, max_size=args.max_size)
        self.frame_height, self.frame_width = self.sdk.source_info["img_rgb_lst"][0].shape[:2]
        self.audio_thread = threading.Thread(
            target=self._guard_worker, args=(self._drive_audio,), daemon=True
        )
        self.motion_thread = threading.Thread(
            target=self._guard_worker, args=(self._drive_motion,), daemon=True
        )
        self.audio_thread.start()
        self.motion_thread.start()

    def _guard_worker(self, target):
        try:
            target()
        except Exception as error:
            self.worker_error = error
            self.stop_event.set()
            print(f"Ditto realtime worker failed: {error}", flush=True)

    def _publish_frame(self, frame_rgb):
        while not self.stop_event.is_set():
            try:
                self.frames.put(frame_rgb, timeout=0.1)
                return
            except queue.Full:
                continue

    def _drive_audio(self):
        deadline = time.monotonic()
        last_voice_time = None
        while not self.stop_event.is_set():
            try:
                packet = self.audio_packets.get(timeout=0.02)
            except queue.Empty:
                packet = bytes(AUDIO_PACKET_BYTES)
            samples = np.frombuffer(packet, dtype="<i2").astype(np.float32) / 32768.0
            now = time.monotonic()
            if float(np.sqrt(np.mean(samples * samples))) >= SPEECH_RMS_THRESHOLD:
                last_voice_time = now
            if last_voice_time is not None and now - last_voice_time <= SPEECH_HANGOVER_SECONDS:
                self._queue_audio_for_motion(samples)
            deadline += 0.02
            delay = deadline - time.monotonic()
            if delay > 0:
                self.stop_event.wait(delay)
            else:
                deadline = time.monotonic()

    def _queue_audio_for_motion(self, packet_samples):
        if not hasattr(self, "_pending_audio"):
            self._pending_audio = np.zeros((0,), dtype=np.float32)
        self._pending_audio = np.concatenate((self._pending_audio, packet_samples))
        while len(self._pending_audio) >= DITTO_FRAME_SAMPLES:
            frame_samples = self._pending_audio[:DITTO_FRAME_SAMPLES]
            self._pending_audio = self._pending_audio[DITTO_FRAME_SAMPLES:]
            self.motion_audio.put(frame_samples, timeout=1)

    def _drive_motion(self):
        samples = np.zeros((0,), dtype=np.float32)
        base_sample = 0
        total_samples = 0
        chunk_index = 0

        while not self.stop_event.is_set():
            try:
                frame_samples = self.motion_audio.get(timeout=0.5)
            except queue.Empty:
                continue
            samples = np.concatenate((samples, frame_samples))
            total_samples += len(frame_samples)

            while total_samples >= chunk_index * DITTO_STEP_SAMPLES + (
                DITTO_WINDOW_SAMPLES - DITTO_PREFIX_SAMPLES
            ):
                window_start = chunk_index * DITTO_STEP_SAMPLES - DITTO_PREFIX_SAMPLES
                window_end = window_start + DITTO_WINDOW_SAMPLES
                left_pad = max(0, -window_start)
                actual_start = max(0, window_start)
                actual_end = window_end
                start_offset = actual_start - base_sample
                end_offset = actual_end - base_sample
                if start_offset < 0 or end_offset > len(samples):
                    raise RuntimeError("Ditto audio context buffer lost required samples")

                window = np.concatenate(
                    (
                        np.zeros((left_pad,), dtype=np.float32),
                        samples[start_offset:end_offset],
                    )
                )
                if len(window) != DITTO_WINDOW_SAMPLES:
                    raise RuntimeError(
                        f"Ditto expected {DITTO_WINDOW_SAMPLES} samples, received {len(window)}"
                    )

                self.sdk.run_chunk(window, chunksize=(3, 5, 2))
                chunk_index += 1

                next_start = max(
                    0, chunk_index * DITTO_STEP_SAMPLES - DITTO_PREFIX_SAMPLES
                )
                discard = next_start - base_sample
                if discard > 0:
                    samples = samples[discard:]
                    base_sample = next_start

    @staticmethod
    def _recv_exact(connection, size):
        data = bytearray()
        while len(data) < size:
            part = connection.recv(size - len(data))
            if not part:
                return None
            data.extend(part)
        return bytes(data)

    def _receive_audio(self, connection):
        while not self.stop_event.is_set():
            packet = self._recv_exact(connection, AUDIO_PACKET_BYTES)
            if packet is None:
                return
            try:
                self.audio_packets.put(packet, timeout=1)
            except queue.Full:
                raise RuntimeError("LiveTalking audio queue is full")

    def _send_video(self, connection, send_lock):
        while True:
            if self.worker_error is not None:
                try:
                    with send_lock:
                        connection.sendall(struct.pack("!I", 0))
                except OSError:
                    pass
                return
            try:
                frame_rgb = self.frames.get(timeout=0.2)
            except queue.Empty:
                continue
            payload = np.ascontiguousarray(frame_rgb).tobytes()
            try:
                with send_lock:
                    connection.sendall(struct.pack("!I", len(payload)))
                    connection.sendall(payload)
            except OSError:
                return

    def serve_client(self, connection):
        send_lock = threading.Lock()
        try:
            connection.sendall(struct.pack("!II", self.frame_height, self.frame_width))
        except OSError:
            connection.close()
            return
        sender = threading.Thread(
            target=self._send_video, args=(connection, send_lock), daemon=True
        )
        sender.start()
        try:
            try:
                self._receive_audio(connection)
            except OSError:
                pass
        finally:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()
            while True:
                try:
                    self.audio_packets.get_nowait()
                except queue.Empty:
                    break
            while True:
                try:
                    self.frames.get_nowait()
                except queue.Empty:
                    break
            sender.join(timeout=1)


def main():
    parser = argparse.ArgumentParser(description="Persistent Ditto-to-LiveTalking worker")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cfg-pkl", required=True)
    parser.add_argument("--source-path", required=True)
    parser.add_argument("--cpu-threads", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--max-size", type=int, default=1024)
    args = parser.parse_args()

    cpu_threads = max(1, args.cpu_threads)
    os.environ["DITTO_CPU_THREADS"] = str(cpu_threads)
    torch.set_num_threads(cpu_threads)
    torch.set_num_interop_threads(1)
    worker = DittoLiveWorker(args)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(1)
        server.settimeout(0.5)
        print(f"Ditto LiveTalking worker listening on {args.host}:{args.port}", flush=True)
        while True:
            try:
                connection, address = server.accept()
            except socket.timeout:
                if worker.worker_error is not None:
                    raise RuntimeError("Ditto streaming worker failed") from worker.worker_error
                continue
            try:
                command = connection.recv(1)
            except OSError:
                connection.close()
                continue
            if command == b"\x00":
                connection.sendall(b"\x01")
                connection.close()
                continue
            if command != b"\x02":
                connection.close()
                continue
            print(f"LiveTalking connected from {address}", flush=True)
            worker.serve_client(connection)
            if worker.worker_error is not None:
                raise RuntimeError("Ditto streaming worker failed") from worker.worker_error


if __name__ == "__main__":
    main()
