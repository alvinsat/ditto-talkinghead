import logging
import os
import sys

import numpy as np


logger = logging.getLogger(__name__)
_dll_directory_handles = []
_dll_directories_registered = False


def _replace_grid_sample_3d(model, node, input_shape, grid_shape):
    import onnx
    from onnx import helper, numpy_helper

    if len(input_shape) != 5 or len(grid_shape) != 5:
        raise ValueError(f"Expected 5D GridSample inputs, got {input_shape} and {grid_shape}")
    if any(dimension <= 0 for dimension in input_shape + grid_shape):
        raise ValueError("OpenVINO GridSample conversion requires static, positive dimensions")

    attributes = {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}
    mode = attributes.get("mode", b"linear")
    padding_mode = attributes.get("padding_mode", b"zeros")
    align_corners = bool(attributes.get("align_corners", 0))
    if mode not in (b"linear", b"bilinear"):
        raise ValueError(f"3D GridSample conversion does not support interpolation mode {mode!r}")
    if padding_mode != b"zeros":
        raise ValueError(f"3D GridSample conversion does not support padding mode {padding_mode!r}")

    batch, channels, depth, height, width = input_shape
    grid_batch, out_depth, out_height, out_width, coordinate_count = grid_shape
    if batch != grid_batch or coordinate_count != 3:
        raise ValueError(f"Incompatible GridSample input shapes: {input_shape} and {grid_shape}")

    prefix = node.name or node.output[0]
    prefix = "".join(character if character.isalnum() else "_" for character in prefix)
    nodes = []
    initializers = []
    counter = 0

    def name(suffix):
        nonlocal counter
        counter += 1
        return f"{prefix}_trilinear_{counter}_{suffix}"

    def constant(suffix, value, dtype=None):
        tensor_name = name(suffix)
        array = np.asarray(value, dtype=dtype)
        initializers.append(numpy_helper.from_array(array, tensor_name))
        return tensor_name

    def op(op_type, inputs, suffix, outputs=1, **attributes):
        output_names = [name(suffix) for _ in range(outputs)]
        nodes.append(helper.make_node(op_type, inputs, output_names, name=name(f"{suffix}_node"), **attributes))
        return output_names[0] if outputs == 1 else output_names

    def binary(op_type, left, right, suffix):
        return op(op_type, [left, right], suffix)

    grid = node.input[1]
    coord_axes = []
    for axis, suffix in enumerate(("x", "y", "z")):
        index = constant(f"{suffix}_axis", [axis], np.int64)
        gathered = op("Gather", [grid, index], f"grid_{suffix}", axis=4)
        coord_axes.append(op("Squeeze", [gathered, constant(f"{suffix}_squeeze_axis", [4], np.int64)], f"coord_{suffix}"))

    def indices_and_weights(coord, size, axis_suffix):
        one = constant(f"{axis_suffix}_one", 1.0, np.float32)
        coordinate_scale = size - 1 if align_corners else size
        size_value = constant(f"{axis_suffix}_size", float(coordinate_scale), np.float32)
        two = constant(f"{axis_suffix}_two", 2.0, np.float32)
        shifted = binary("Add", coord, one, f"{axis_suffix}_shift")
        scaled = binary("Mul", shifted, size_value, f"{axis_suffix}_scale")
        if not align_corners:
            scaled = binary("Sub", scaled, one, f"{axis_suffix}_align")
        pixel = binary("Div", scaled, two, f"{axis_suffix}_pixel")
        lower_float = op("Floor", [pixel], f"{axis_suffix}_floor")
        lower = op("Cast", [lower_float], f"{axis_suffix}_lower", to=onnx.TensorProto.INT64)
        upper = binary("Add", lower, constant(f"{axis_suffix}_one_i64", 1, np.int64), f"{axis_suffix}_upper")
        upper_float = op("Cast", [upper], f"{axis_suffix}_upper_float", to=onnx.TensorProto.FLOAT)
        lower_weight = binary("Sub", upper_float, pixel, f"{axis_suffix}_lower_weight")
        upper_weight = binary("Sub", pixel, lower_float, f"{axis_suffix}_upper_weight")
        size_i64 = constant(f"{axis_suffix}_size_i64", size, np.int64)
        zero_i64 = constant(f"{axis_suffix}_zero_i64", 0, np.int64)

        def clip_and_mask(index, side):
            lower_bound = binary("GreaterOrEqual", index, zero_i64, f"{axis_suffix}_{side}_gte_zero")
            upper_bound = binary("Less", index, size_i64, f"{axis_suffix}_{side}_lt_size")
            valid = binary("And", lower_bound, upper_bound, f"{axis_suffix}_{side}_valid")
            clipped_low = op("Max", [index, zero_i64], f"{axis_suffix}_{side}_clip_low")
            clipped = op("Min", [clipped_low, constant(f"{axis_suffix}_{side}_max", size - 1, np.int64)], f"{axis_suffix}_{side}_clip")
            mask = op("Cast", [valid], f"{axis_suffix}_{side}_mask", to=onnx.TensorProto.FLOAT)
            return clipped, mask

        lower, lower_mask = clip_and_mask(lower, "lower")
        upper, upper_mask = clip_and_mask(upper, "upper")
        return (
            (lower, lower_weight, lower_mask),
            (upper, upper_weight, upper_mask),
        )

    x_indices = indices_and_weights(coord_axes[0], width, "x")
    y_indices = indices_and_weights(coord_axes[1], height, "y")
    z_indices = indices_and_weights(coord_axes[2], depth, "z")

    data_n_d_h_w_c = op("Transpose", [node.input[0]], "data_n_d_h_w_c", perm=[0, 2, 3, 4, 1])
    flat_shape = constant("flat_shape", [batch, depth * height * width, channels], np.int64)
    data_flat = op("Reshape", [data_n_d_h_w_c, flat_shape], "data_flat")
    point_count = out_depth * out_height * out_width
    grid_shape_constant = constant("grid_shape", [batch, point_count], np.int64)
    output_shape_constant = constant(
        "output_shape", [batch, out_depth, out_height, out_width, channels], np.int64
    )
    channel_axis = constant("channel_axis", [4], np.int64)
    output_permutation = [0, 4, 1, 2, 3]
    samples = []

    for z_index, z_weight, z_mask in z_indices:
        for y_index, y_weight, y_mask in y_indices:
            for x_index, x_weight, x_mask in x_indices:
                y_times_width = binary("Mul", y_index, constant("width_i64", width, np.int64), "linear_y")
                z_times_plane = binary(
                    "Mul",
                    z_index,
                    constant("plane_i64", height * width, np.int64),
                    "linear_z",
                )
                flat_index = binary("Add", binary("Add", z_times_plane, y_times_width, "linear_zy"), x_index, "flat_index")
                flat_index = op("Reshape", [flat_index, grid_shape_constant], "flat_index_2d")
                flat_index = op("Unsqueeze", [flat_index, constant("index_axis", [2], np.int64)], "gather_indices")
                sample = op(
                    "GatherND",
                    [data_flat, flat_index],
                    "gather",
                    batch_dims=1,
                )

                weight = binary("Mul", x_weight, y_weight, "weight_xy")
                weight = binary("Mul", weight, z_weight, "weight_xyz")
                mask = binary("Mul", x_mask, y_mask, "mask_xy")
                mask = binary("Mul", mask, z_mask, "mask_xyz")
                weight = binary("Mul", weight, mask, "weight_masked")
                weight = op("Reshape", [weight, grid_shape_constant], "weight_2d")
                weight = op("Unsqueeze", [weight, constant("weight_axis", [2], np.int64)], "weight_3d")
                samples.append(binary("Mul", sample, weight, "weighted_sample"))

    result = samples[0]
    for sample in samples[1:]:
        result = binary("Add", result, sample, "sum_samples")

    result = op(
        "Reshape",
        [result, output_shape_constant],
        "sample_n_d_h_w_c",
    )
    result = op("Transpose", [result], "sample_n_c_d_h_w", perm=output_permutation)
    result_nodes = nodes
    result_nodes[-1].output[0] = node.output[0]
    return result_nodes, initializers


def convert_5d_gridsample_model(source_path, output_path):
    import onnx
    from onnx import numpy_helper

    model = onnx.load(source_path)
    shape_by_name = {}
    for value in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        if not value.type.HasField("tensor_type"):
            continue
        tensor_shape = value.type.tensor_type.shape
        dimensions = []
        for dimension in tensor_shape.dim:
            if not dimension.HasField("dim_value"):
                dimensions.append(0)
            else:
                dimensions.append(dimension.dim_value)
        shape_by_name[value.name] = dimensions

    initializers_by_name = {tensor.name: tensor for tensor in model.graph.initializer}
    producer_by_output = {
        output: node for node in model.graph.node for output in node.output
    }
    remove_nodes = set()
    rewritten_value_names = set()
    for node in model.graph.node:
        if node.op_type != "Tile" or len(node.input) != 2:
            continue
        unsqueeze = producer_by_output.get(node.input[0])
        repeats_tensor = initializers_by_name.get(node.input[1])
        if unsqueeze is None or unsqueeze.op_type != "Unsqueeze" or repeats_tensor is None:
            continue
        repeats = numpy_helper.to_array(repeats_tensor)
        source_shape = shape_by_name.get(unsqueeze.input[0], [])
        if (
            len(repeats) == 7
            and repeats[0] == 1
            and repeats[1] > 1
            and np.all(repeats[2:] == 1)
            and len(source_shape) == 5
        ):
            node.input[0] = unsqueeze.input[0]
            repeats_tensor.CopyFrom(
                numpy_helper.from_array(
                    np.asarray([repeats[1], 1, 1, 1, 1], dtype=repeats.dtype),
                    node.input[1],
                )
            )
            remove_nodes.add(unsqueeze.name)
            rewritten_value_names.update(unsqueeze.output)
            rewritten_value_names.update(node.output)
            logger.info("Reduced 7D Tile %s to a 5D Tile for OpenVINO GPU", node.name)

    if remove_nodes:
        retained_nodes = [node for node in model.graph.node if node.name not in remove_nodes]
        del model.graph.node[:]
        model.graph.node.extend(retained_nodes)
        retained_value_info = [
            value for value in model.graph.value_info
            if value.name not in rewritten_value_names
        ]
        del model.graph.value_info[:]
        model.graph.value_info.extend(retained_value_info)

    converted_nodes = []
    converted_initializers = []
    converted_count = 0
    for node in model.graph.node:
        if node.op_type != "GridSample":
            converted_nodes.append(node)
            continue

        input_shape = shape_by_name.get(node.input[0])
        grid_shape = shape_by_name.get(node.input[1])
        if input_shape is None or grid_shape is None:
            raise ValueError(f"Could not determine static GridSample shapes for {node.name}")
        if len(input_shape) != 5:
            converted_nodes.append(node)
            continue

        replacement, initializers = _replace_grid_sample_3d(model, node, input_shape, grid_shape)
        converted_nodes.extend(replacement)
        converted_initializers.extend(initializers)
        converted_count += 1

    if converted_count == 0:
        raise ValueError(f"No 5D GridSample nodes were found in {source_path}")

    del model.graph.node[:]
    model.graph.node.extend(converted_nodes)
    model.graph.initializer.extend(converted_initializers)
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    logger.info("Converted %d 5D GridSample nodes into ONNX ops in %s", converted_count, output_path)
    return converted_count


def get_openvino_warp_model_path(model_path):
    output_path = os.path.splitext(model_path)[0] + "_openvino.onnx"
    if not os.path.isfile(output_path) or os.path.getmtime(output_path) < os.path.getmtime(model_path):
        convert_5d_gridsample_model(model_path, output_path)
    return output_path


def normalize_gridsample_modes(model_path):
    import onnx

    model = onnx.load(model_path)
    modified_modes = []
    for node in model.graph.node:
        if node.op_type != "GridSample":
            continue
        for attribute in node.attribute:
            if attribute.name == "mode" and attribute.s in (b"bilinear", b"bicubic"):
                old_mode = attribute.s.decode("ascii")
                attribute.s = b"linear" if attribute.s == b"bilinear" else b"cubic"
                modified_modes.append(f"{old_mode}->{attribute.s.decode('ascii')}")

    if modified_modes:
        onnx.save(model, model_path)
        logger.info(
            "Normalized GridSample modes in %s: %s",
            model_path,
            ", ".join(modified_modes),
        )

    return modified_modes


def _register_conda_dll_directories():
    global _dll_directories_registered

    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    if _dll_directories_registered:
        return

    conda_prefix = sys.prefix
    candidates = (
        os.path.join(conda_prefix, "Lib", "site-packages", "openvino", "libs"),
        os.path.join(conda_prefix, "Library", "bin"),
    )
    for dll_directory in candidates:
        if os.path.isdir(dll_directory):
            _dll_directory_handles.append(os.add_dll_directory(dll_directory))
            os.environ["PATH"] = dll_directory + os.pathsep + os.environ.get("PATH", "")
    _dll_directories_registered = True


def _provider_options(provider_name):
    if provider_name == "OpenVINOExecutionProvider":
        return (
            provider_name,
            {
                "device_type": os.environ.get("DITTO_OPENVINO_DEVICE", "GPU"),
                "precision": os.environ.get("DITTO_OPENVINO_PRECISION", "FP16"),
            },
        )
    return provider_name


def create_onnx_session(model_path, providers=None):
    _register_conda_dll_directories()
    import onnxruntime

    cpu_threads = max(1, int(os.environ.get("DITTO_CPU_THREADS", "4")))
    session_options = onnxruntime.SessionOptions()
    session_options.intra_op_num_threads = cpu_threads
    session_options.inter_op_num_threads = 1
    session_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL

    available = set(onnxruntime.get_available_providers())
    if providers is not None:
        requested = providers
        attempts = [requested]
    else:
        priority = (
            "OpenVINOExecutionProvider",
            "CUDAExecutionProvider",
            "DmlExecutionProvider",
            "CPUExecutionProvider",
        )
        ordered = [name for name in priority if name in available]
        if not ordered:
            raise RuntimeError(
                f"No supported ONNX Runtime execution providers are available: "
                f"{onnxruntime.get_available_providers()}"
            )

        attempts = [[_provider_options(name) for name in ordered]]
        for index, name in enumerate(ordered[:-1]):
            fallback = ordered[index + 1 :]
            attempts.append([_provider_options(name) for name in fallback])

    errors = []
    for attempt in attempts:
        attempt_names = [
            provider[0] if isinstance(provider, tuple) else provider
            for provider in attempt
        ]
        try:
            session = onnxruntime.InferenceSession(
                model_path,
                sess_options=session_options,
                providers=attempt,
            )
            logger.info(
                "Loaded ONNX model %s with execution providers %s",
                model_path,
                session.get_providers(),
            )
            return session
        except Exception as error:
            errors.append((attempt_names, error))
            if len(attempts) > 1:
                logger.warning(
                    "Could not load ONNX model %s with providers %s; trying fallback: %s",
                    model_path,
                    attempt_names,
                    error,
                )

    attempts_description = "; ".join(
        f"{names}: {error}" for names, error in errors
    )
    raise RuntimeError(
        f"Could not load ONNX model {model_path} with any configured provider. "
        f"Attempts: {attempts_description}"
    ) from errors[-1][1]
