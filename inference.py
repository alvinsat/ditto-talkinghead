import librosa
import math
import os
import numpy as np
import random
import subprocess
import torch
import pickle
import time
from tqdm import tqdm

from stream_pipeline_offline import StreamSDK


def _log_progress(message, started_at):
    elapsed = time.monotonic() - started_at
    print(f"[{elapsed:7.1f}s] {message}", flush=True)


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pkl(pkl):
    with open(pkl, "rb") as f:
        return pickle.load(f)


def run(SDK: StreamSDK, audio_path: str, source_path: str, output_path: str, more_kwargs: str | dict = {}):
    started_at = time.monotonic()
    _log_progress("Preparing inference options", started_at)

    if isinstance(more_kwargs, str):
        more_kwargs = load_pkl(more_kwargs)
    setup_kwargs = more_kwargs.get("setup_kwargs", {})
    run_kwargs = more_kwargs.get("run_kwargs", {})

    _log_progress("Registering source and preparing rendering pipeline", started_at)
    SDK.setup(source_path, output_path, **setup_kwargs)
    _log_progress("Source registered; loading audio at 16 kHz", started_at)

    audio, sr = librosa.core.load(audio_path, sr=16000)
    num_f = math.ceil(len(audio) / 16000 * 25)
    _log_progress(
        f"Audio loaded ({len(audio) / 16000:.1f}s); generating {num_f} output frames",
        started_at,
    )

    fade_in = run_kwargs.get("fade_in", -1)
    fade_out = run_kwargs.get("fade_out", -1)
    ctrl_info = run_kwargs.get("ctrl_info", {})
    SDK.setup_Nd(N_d=num_f, fade_in=fade_in, fade_out=fade_out, ctrl_info=ctrl_info)

    online_mode = SDK.online_mode
    if online_mode:
        chunksize = run_kwargs.get("chunksize", (3, 5, 2))
        audio = np.concatenate([np.zeros((chunksize[0] * 640,), dtype=np.float32), audio], 0)
        split_len = int(sum(chunksize) * 0.04 * 16000) + 80  # 6480
        chunk_starts = range(0, len(audio), chunksize[1] * 640)
        _log_progress(f"Streaming audio in {len(chunk_starts)} chunks", started_at)
        for i in tqdm(chunk_starts, desc="audio", unit="chunk"):
            audio_chunk = audio[i:i + split_len]
            if len(audio_chunk) < split_len:
                audio_chunk = np.pad(audio_chunk, (0, split_len - len(audio_chunk)), mode="constant")
            SDK.run_chunk(audio_chunk, chunksize)
    else:
        _log_progress("Extracting HuBERT audio features", started_at)
        aud_feat = SDK.wav2feat.wav2feat(audio)
        _log_progress(f"Audio features ready ({len(aud_feat)} feature frames)", started_at)
        SDK.audio2motion_queue.put(aud_feat)
    _log_progress("Waiting for motion and rendering workers to finish", started_at)
    SDK.close()
    _log_progress("Rendered frames; muxing source audio into the final MP4", started_at)

    subprocess.run(
        [
            "ffmpeg", "-loglevel", "error", "-y",
            "-i", SDK.tmp_output_path,
            "-i", audio_path,
            "-map", "0:v", "-map", "1:a",
            "-c:v", "copy", "-c:a", "aac",
            output_path,
        ],
        check=True,
    )

    _log_progress(f"Finished: {output_path}", started_at)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./checkpoints/ditto_trt_Ampere_Plus", help="path to trt data_root")
    parser.add_argument("--cfg_pkl", type=str, default="./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl", help="path to cfg_pkl")

    parser.add_argument("--audio_path", type=str, help="path to input wav")
    parser.add_argument("--source_path", type=str, help="path to input image")
    parser.add_argument("--output_path", type=str, help="path to output mp4")
    args = parser.parse_args()

    SDK = None
    try:
        data_root = args.data_root
        cfg_pkl = args.cfg_pkl
        started_at = time.monotonic()
        _log_progress(f"Loading model sessions from {data_root}", started_at)
        SDK = StreamSDK(cfg_pkl, data_root)
        _log_progress("Model sessions loaded", started_at)
        run(SDK, args.audio_path, args.source_path, args.output_path)
    except KeyboardInterrupt:
        print("\nCancellation requested; stopping inference workers...", flush=True)
        if SDK is not None:
            SDK.cancel()
        print("Inference cancelled. Partial temporary video, if any, was left at the .tmp.mp4 path.", flush=True)
        raise SystemExit(130) from None
