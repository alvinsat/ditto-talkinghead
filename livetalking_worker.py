import argparse
import queue
import socket
import struct
import threading
import time
import os

import numpy as np
import torch
import cv2

from stream_pipeline_online import StreamSDK


AUDIO_PACKET_SAMPLES = 320
AUDIO_PACKET_BYTES = AUDIO_PACKET_SAMPLES * 2
DITTO_FRAME_SAMPLES = 640
DITTO_STEP_SAMPLES = 5 * DITTO_FRAME_SAMPLES
DITTO_PREFIX_SAMPLES = 2000
DITTO_WINDOW_SAMPLES = 6480
GENERATED_FRAME_BUFFER_SIZE = 16
END_UTTERANCE_FRAME_LENGTH = 0xFFFFFFFF


class _EndUtterance:
    def __init__(self, completed=None, expected_frames=None):
        self.completed = completed
        self.expected_frames = expected_frames


END_UTTERANCE = _EndUtterance()
CLOSE_CLIENT = object()


class DittoLiveWorker:
    def __init__(self, args):
        self.frames = queue.Queue(maxsize=GENERATED_FRAME_BUFFER_SIZE)
        self.audio_packets = queue.Queue()
        self.motion_audio = queue.Queue()
        self.worker_error = None
        self.stop_event = threading.Event()
        self.sdk = StreamSDK(
            args.cfg_pkl,
            args.data_root,
            frame_sink=self._publish_frame,
            putback_mask_mode=args.patch_region,
        )
        self.sdk.setup(
            args.source_path,
            output_path=None,
            max_size=args.max_size,
            sampling_timesteps=args.sampling_timesteps,
        )
        self.sdk.run_chunk(
            np.zeros((DITTO_WINDOW_SAMPLES,), dtype=np.float32),
            chunksize=(3, 5, 2),
        )
        self.sdk.flush()
        self.sdk.begin_progress()
        while True:
            try:
                self.frames.get_nowait()
            except queue.Empty:
                break
        self.frame_height, self.frame_width = self.sdk.source_info["img_rgb_lst"][0].shape[:2]
        self._fps_window_start = time.monotonic()
        self._fps_window_frames = 0
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
                ok, encoded = cv2.imencode(
                    ".jpg",
                    cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 85],
                )
                if not ok:
                    raise RuntimeError("Could not encode Ditto output frame")
                self.frames.put(encoded.tobytes(), timeout=0.1)
                self._fps_window_frames += 1
                if self._fps_window_frames >= 100:
                    now = time.monotonic()
                    elapsed = max(now - self._fps_window_start, 1e-9)
                    writer_current, writer_total = self.sdk.get_progress()
                    writer_total_text = (
                        str(writer_total) if writer_total is not None else "?"
                    )
                    print(
                        "Ditto generated-frame throughput: "
                        f"{self._fps_window_frames / elapsed:.2f} fps "
                        f"({self._fps_window_frames} frames / {elapsed:.2f} s; "
                        f"writer progress {writer_current}/{writer_total_text} frames)",
                        flush=True,
                    )
                    self._fps_window_start = now
                    self._fps_window_frames = 0
                return
            except queue.Full:
                continue

    def _drive_audio(self):
        pending_samples = np.zeros((0,), dtype=np.float32)
        utterance_frame_count = 0
        progress_started = False
        while not self.stop_event.is_set():
            try:
                packet = self.audio_packets.get(timeout=0.2)
            except queue.Empty:
                continue

            if isinstance(packet, _EndUtterance):
                if packet.expected_frames is not None:
                    self.sdk.set_expected_frames(packet.expected_frames)
                if len(pending_samples):
                    frame_samples = np.pad(
                        pending_samples,
                        (0, DITTO_FRAME_SAMPLES - len(pending_samples)),
                    )
                    self.motion_audio.put(frame_samples)
                    utterance_frame_count += 1
                    pending_samples = np.zeros((0,), dtype=np.float32)
                while utterance_frame_count % 5:
                    self.motion_audio.put(
                        np.zeros((DITTO_FRAME_SAMPLES,), dtype=np.float32)
                    )
                    utterance_frame_count += 1
                self.motion_audio.put(packet)
                utterance_frame_count = 0
                progress_started = False
                continue

            if not progress_started:
                self.sdk.begin_progress()
                progress_started = True
            samples = np.frombuffer(packet, dtype="<i2").astype(np.float32) / 32768.0
            pending_samples = np.concatenate((pending_samples, samples))
            while len(pending_samples) >= DITTO_FRAME_SAMPLES:
                frame_samples = pending_samples[:DITTO_FRAME_SAMPLES]
                pending_samples = pending_samples[DITTO_FRAME_SAMPLES:]
                self.motion_audio.put(frame_samples)
                utterance_frame_count += 1

    def _drive_motion(self):
        samples = np.zeros((0,), dtype=np.float32)
        base_sample = 0
        total_samples = 0
        chunk_index = 0
        utterance_sample_count = 0

        while not self.stop_event.is_set():
            try:
                frame_samples = self.motion_audio.get(timeout=0.5)
            except queue.Empty:
                continue
            if isinstance(frame_samples, _EndUtterance):
                if utterance_sample_count:
                    while total_samples < chunk_index * DITTO_STEP_SAMPLES + (
                        DITTO_WINDOW_SAMPLES - DITTO_PREFIX_SAMPLES
                    ):
                        tail = np.zeros((DITTO_FRAME_SAMPLES,), dtype=np.float32)
                        samples = np.concatenate((samples, tail))
                        total_samples += len(tail)
            else:
                samples = np.concatenate((samples, frame_samples))
                total_samples += len(frame_samples)
                utterance_sample_count += len(frame_samples)

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

            if isinstance(frame_samples, _EndUtterance):
                self.sdk.flush(finalize_progress=True)
                self.frames.put(END_UTTERANCE, timeout=1)
                if frame_samples.completed is not None:
                    self.frames.put(CLOSE_CLIENT, timeout=1)
                    frame_samples.completed.set()
                utterance_sample_count = 0

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
        received_audio = False
        utterance_packet_count = 0
        while not self.stop_event.is_set():
            try:
                message_type = self._recv_exact(connection, 1)
            except OSError:
                return received_audio
            if message_type is None:
                return received_audio
            if message_type == b"\x01":
                try:
                    packet = self._recv_exact(connection, AUDIO_PACKET_BYTES)
                except OSError:
                    return received_audio
                if packet is None:
                    return received_audio
                received_audio = True
                utterance_packet_count += 1
            elif message_type == b"\x02":
                expected_frames = (
                    ((utterance_packet_count + 1) // 2 + 4) // 5
                ) * 5
                packet = _EndUtterance(expected_frames=expected_frames)
                utterance_packet_count = 0
            else:
                raise RuntimeError(
                    f"Unknown LiveTalking audio message: {message_type!r}"
                )
            self.audio_packets.put(packet)

    def _send_video(self, connection, send_lock):
        connection_failed = False
        while True:
            if self.worker_error is not None:
                if not connection_failed:
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
            if frame_rgb is CLOSE_CLIENT:
                return
            if frame_rgb is END_UTTERANCE:
                if connection_failed:
                    continue
                try:
                    with send_lock:
                        connection.sendall(
                            struct.pack("!I", END_UTTERANCE_FRAME_LENGTH)
                        )
                except OSError:
                    connection_failed = True
                continue
            if connection_failed:
                continue
            payload = frame_rgb
            try:
                with send_lock:
                    connection.sendall(struct.pack("!I", len(payload)))
                    connection.sendall(payload)
            except OSError:
                connection_failed = True

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
        received_audio = False
        try:
            try:
                received_audio = self._receive_audio(connection)
            except OSError:
                pass
        finally:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()
            if received_audio:
                completion = threading.Event()
                disconnect_end = _EndUtterance(completion)
                deadline = time.monotonic() + 120
                while time.monotonic() < deadline and not self.stop_event.is_set():
                    try:
                        self.audio_packets.put(disconnect_end, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                while not completion.is_set() and not self.stop_event.is_set():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    completion.wait(min(0.1, remaining))
                if not completion.is_set() and not self.stop_event.is_set():
                    self.worker_error = TimeoutError(
                        "Ditto worker timed out draining a disconnected utterance"
                    )
                    print(
                        "Ditto worker timed out draining a disconnected utterance",
                        flush=True,
                    )
            else:
                self.frames.put(CLOSE_CLIENT)
            sender.join(timeout=2)
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


def main():
    parser = argparse.ArgumentParser(description="Persistent Ditto-to-LiveTalking worker")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cfg-pkl", required=True)
    parser.add_argument("--source-path", required=True)
    parser.add_argument("--sampling-timesteps", type=int, default=15)
    parser.add_argument(
        "--patch-region",
        choices=("face", "lower_face"),
        default="lower_face",
    )
    parser.add_argument("--cpu-threads", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--max-size", type=int, default=1024)
    args = parser.parse_args()
    if args.sampling_timesteps < 1:
        parser.error("--sampling-timesteps must be at least 1")

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
