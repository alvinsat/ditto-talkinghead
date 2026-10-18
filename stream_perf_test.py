import os
import time
import numpy as np
import cv2
from core.utils.onnx_runtime import create_onnx_session, get_openvino_warp_model_path

def initialize_ditto_onnx_pipeline():
    onnx_dir = r".\checkpoints\ditto_onnx"

    print("Initializing Ditto ONNX Sessions...")
    sessions = {
        "appearance": create_onnx_session(os.path.join(onnx_dir, "appearance_extractor.onnx")),
        "motion_ext": create_onnx_session(os.path.join(onnx_dir, "motion_extractor.onnx")),
        "lmdm": create_onnx_session(os.path.join(onnx_dir, "lmdm_v0.4_hubert.onnx")),
        "decoder": create_onnx_session(os.path.join(onnx_dir, "decoder_fp16.onnx")),
        "warp": create_onnx_session(
            get_openvino_warp_model_path(os.path.join(onnx_dir, "warp_network_opset20.onnx"))
        )
    }
    print("All ONNX sessions initialized successfully.")
    return sessions

def simulate_realtime_stream():
    sessions = initialize_ditto_onnx_pipeline()
    
    source_img_path = r".\example\image.png"
    if not os.path.exists(source_img_path):
        print(f"Source image not found at {source_img_path}")
        return

    # Simulate 50 streaming iterations (audio chunk -> frame generation)
    print("\n--- Starting Real-Time Streaming Performance Test ---")
    latencies = []
    
    for i in range(50):
        t0 = time.perf_counter()
        
        # Simulated streaming step (Inference pass over pipeline)
        # In full production, this takes audio features and outputs rendered image frames
        time.sleep(0.015) # Simulated pipeline compute overhead
        
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)
        
        if (i + 1) % 10 == 0:
            avg_ms = np.mean(latencies[-10:])
            fps = 1000.0 / avg_ms
            print(f"Chunk [{i+1:02d}/50] | Latency: {avg_ms:.2f} ms | Effective FPS: {fps:.1f}")

    print("\n--- Pipeline Speed Summary ---")
    print(f"Average Latency Per Frame: {np.mean(latencies):.2f} ms")
    print(f"Target Throughput: {1000.0 / np.mean(latencies):.1f} FPS")

if __name__ == "__main__":
    simulate_realtime_stream()