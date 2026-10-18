
from ...utils.onnx_runtime import create_onnx_session


class HubertStreamingONNX:
    def __init__(self, model_file, device="cuda"):
        self.session = create_onnx_session(model_file)

    def forward_chunk(self, input_values):
        encoding_out = self.session.run(
            None,
            {"input_values": input_values.reshape(1, -1)}
        )[0]
        return encoding_out
    
