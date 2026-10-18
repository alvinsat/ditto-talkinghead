import os
from core.utils.onnx_runtime import create_onnx_session, get_openvino_warp_model_path


class DittoIntelArcPipeline:
    def __init__(self, onnx_dir=r".\checkpoints\ditto_onnx"):
        self.onnx_dir = onnx_dir

        print("--- Loading Ditto ONNX Sub-modules ---")
        self.sess_appearance = create_onnx_session(os.path.join(onnx_dir, "appearance_extractor.onnx"))
        self.sess_motion_ext = create_onnx_session(os.path.join(onnx_dir, "motion_extractor.onnx"))
        self.sess_hubert = create_onnx_session(os.path.join(onnx_dir, "hubert.onnx"))
        self.sess_lmdm = create_onnx_session(os.path.join(onnx_dir, "lmdm_v0.4_hubert.onnx"))
        self.sess_decoder = create_onnx_session(os.path.join(onnx_dir, "decoder_fp16.onnx"))
        warp_model_path = get_openvino_warp_model_path(
            os.path.join(onnx_dir, "warp_network_opset20.onnx")
        )
        self.sess_warp = create_onnx_session(warp_model_path)
        print("All sessions loaded successfully!\n")

    def inspect_inputs(self):
        print("--- Input Specifications ---")
        print("Decoder Input 0:", self.sess_decoder.get_inputs()[0].name, self.sess_decoder.get_inputs()[0].shape)
        print("Warp Input 0:   ", self.sess_warp.get_inputs()[0].name, self.sess_warp.get_inputs()[0].shape)

if __name__ == "__main__":
    pipeline = DittoIntelArcPipeline()
    pipeline.inspect_inputs()