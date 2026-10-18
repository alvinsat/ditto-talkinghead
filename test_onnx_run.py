import glob
import os

from core.utils.onnx_runtime import create_onnx_session, get_openvino_warp_model_path, normalize_gridsample_modes


def fix_gridsample_modes(onnx_dir):
    onnx_files = glob.glob(os.path.join(onnx_dir, "*.onnx"))
    print(f"Scanning {len(onnx_files)} ONNX models for CPU-incompatible GridSample mode names...")

    for file_path in onnx_files:
        if os.path.basename(file_path) == "warp_network_opset20.onnx":
            continue
        modified_modes = normalize_gridsample_modes(file_path)
        if modified_modes:
            print(f"[PATCHED] {os.path.basename(file_path)}: {', '.join(modified_modes)}")


def run_onnx_gpu_test():
    onnx_dir = r".\checkpoints\ditto_onnx"
    gpu_provider = [
        (
            "OpenVINOExecutionProvider",
            {
                "device_type": "GPU",
                "precision": os.environ.get("DITTO_OPENVINO_PRECISION", "FP16"),
            },
        )
    ]

    fix_gridsample_modes(onnx_dir)

    models_to_test = {
        "appearance_extractor.onnx": gpu_provider,
        "motion_extractor.onnx": gpu_provider,
        "hubert.onnx": gpu_provider,
        "lmdm_v0.4_hubert.onnx": gpu_provider,
        "decoder.onnx": gpu_provider,
        "stitch_network.onnx": gpu_provider,
        "warp_network_ori.onnx": gpu_provider,
        "blaze_face.onnx": gpu_provider,
        "face_mesh.onnx": gpu_provider,
        "insightface_det.onnx": gpu_provider,
        "landmark106.onnx": gpu_provider,
        "landmark203.onnx": gpu_provider,
    }

    print("--- Testing Ditto ONNX Model Loading & Execution Providers ---")
    failures = []

    for model_file, providers in models_to_test.items():
        model_path = os.path.join(onnx_dir, model_file)
        if model_file == "warp_network_ori.onnx":
            model_path = get_openvino_warp_model_path(model_path)
        if not os.path.exists(model_path):
            print(f"[MISSING] {model_file} not found in {onnx_dir}")
            failures.append(model_file)
            continue

        try:
            session = create_onnx_session(model_path, providers=providers)
            active_providers = session.get_providers()
            if "OpenVINOExecutionProvider" not in active_providers:
                raise RuntimeError(f"OpenVINO GPU provider not active: {active_providers}")
            print(f"[SUCCESS] Loaded {model_file:<25} | Provider: OpenVINO GPU")
        except Exception as e:
            print(f"[FAIL] Could not load {model_file}: {e}")
            failures.append(model_file)

    if failures:
        raise SystemExit(f"ONNX smoke test failed for: {', '.join(failures)}")


if __name__ == "__main__":
    run_onnx_gpu_test()