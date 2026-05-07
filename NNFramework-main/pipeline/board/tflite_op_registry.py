from __future__ import annotations


# Resolver lines shared by firmware code generation and support summaries.
SUPPORTED_TFLITE_RESOLVER_LINES: dict[str, str] = {
    "ADD": "resolver.AddAdd();",
    "AVERAGE_POOL_2D": "resolver.AddAveragePool2D();",
    "CONCATENATION": "resolver.AddConcatenation();",
    "CONV_2D": "resolver.AddConv2D();",
    "DEQUANTIZE": "resolver.AddDequantize();",
    "DEPTHWISE_CONV_2D": "resolver.AddDepthwiseConv2D();",
    "FULLY_CONNECTED": "resolver.AddFullyConnected();",
    "LOGISTIC": "resolver.AddLogistic();",
    "MAX_POOL_2D": "resolver.AddMaxPool2D();",
    "MEAN": "resolver.AddMean();",
    "MUL": "resolver.AddMul();",
    "PAD": "resolver.AddPad();",
    "QUANTIZE": "resolver.AddQuantize();",
    "RELU": "resolver.AddRelu();",
    "RESHAPE": "resolver.AddReshape();",
    "SOFTMAX": "resolver.AddSoftmax();",
    "STRIDED_SLICE": "resolver.AddStridedSlice();",
}


def build_resolver_lines(op_names: list[str]) -> list[str]:
    lines = []
    for op_name in op_names:
        resolver_line = SUPPORTED_TFLITE_RESOLVER_LINES.get(op_name)
        if resolver_line is None:
            raise ValueError(
                f"Unsupported TFLite operator for generated nRF5340 runner: {op_name}"
            )
        lines.append(resolver_line)
    return lines
