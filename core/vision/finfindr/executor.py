"""Small inference-only executor for the two pinned FinFindR MXNet graphs.

It is not a general MXNet reimplementation. Every supported operator is here
because it occurs in the exact 0.1.21 crop or six-class edge checkpoint.
Unknown operators fail closed.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, List, Mapping, Sequence, Tuple, Union

import numpy as np

from .checkpoint import read_mxnet_params
from .errors import FinFindRDependencyError, FinFindRModelError


def _torch_modules():
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:
        raise FinFindRDependencyError(
            "FinFindR inference requires PyTorch. Install the optional dependencies "
            "with: pip install -e '.[finfindr]'"
        ) from exc
    return torch, functional


def _literal(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def _bool_attr(attrs: Mapping[str, str], name: str, default: bool = False) -> bool:
    value = attrs.get(name)
    if value is None:
        return default
    return str(value).lower() in {"1", "true", "yes"}


def _tuple_attr(
    attrs: Mapping[str, str], name: str, default: Sequence[int]
) -> Tuple[int, ...]:
    value = _literal(attrs.get(name), tuple(default))
    if isinstance(value, (int, float)):
        value = (value,)
    return tuple(int(item) for item in value)


class MXNetSymbolExecutor:
    """Execute a pinned MXNet symbol graph using inference-only PyTorch ops."""

    def __init__(self, symbol_path: Path, params_path: Path, device: str = "cpu"):
        torch, _functional = _torch_modules()
        self.torch = torch
        self.device = torch.device(device)
        if self.device.type != "cpu" and not (
            self.device.type == "mps" and torch.backends.mps.is_available()
        ):
            raise FinFindRDependencyError(f"Requested inference device is unavailable: {device}.")
        try:
            self.graph = json.loads(Path(symbol_path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise FinFindRModelError(f"Cannot read MXNet symbol graph: {symbol_path}") from exc
        self.nodes = self.graph.get("nodes")
        self.heads = self.graph.get("heads")
        if not isinstance(self.nodes, list) or not isinstance(self.heads, list):
            raise FinFindRModelError("MXNet symbol graph is missing nodes or heads.")
        arrays = read_mxnet_params(Path(params_path))
        self.params = {
            name: torch.from_numpy(np.asarray(value, dtype=np.float32)).to(self.device)
            for name, value in arrays.items()
        }
        self._validate_graph_inputs()

    def _validate_graph_inputs(self) -> None:
        null_names = {node.get("name") for node in self.nodes if node.get("op") == "null"}
        missing = sorted(name for name in null_names if name != "data" and name not in self.params)
        if missing:
            raise FinFindRModelError(
                "Checkpoint does not contain symbol inputs: " + ", ".join(missing[:10])
            )

    def predict(self, image_nchw: np.ndarray) -> np.ndarray:
        array = np.asarray(image_nchw, dtype=np.float32)
        if array.ndim != 4 or array.shape[0] != 1 or array.shape[1] != 3:
            raise ValueError(f"Expected input shape (1, 3, height, width), got {array.shape}.")
        data = self.torch.from_numpy(np.ascontiguousarray(array)).to(self.device)
        with self.torch.inference_mode():
            values = self._execute(data)
        if len(values) != 1:
            raise FinFindRModelError(f"Expected one model output, got {len(values)}.")
        return values[0].detach().to("cpu").numpy().astype(np.float32, copy=False)

    def _execute(self, data):
        results: List[Union[Any, Tuple[Any, ...]]] = []
        for index, node in enumerate(self.nodes):
            inputs = [self._resolve(results, ref) for ref in node.get("inputs", [])]
            try:
                value = self._operation(node, inputs, data)
            except FinFindRModelError:
                raise
            except Exception as exc:
                raise FinFindRModelError(
                    f"FinFindR graph failed at node {index} "
                    f"{node.get('name')!r} ({node.get('op')!r}): {exc}"
                ) from exc
            results.append(value)
        return [self._resolve(results, ref) for ref in self.heads]

    @staticmethod
    def _resolve(results, reference):
        node_index, output_index = int(reference[0]), int(reference[1])
        value = results[node_index]
        if isinstance(value, tuple):
            return value[output_index]
        if output_index != 0:
            raise FinFindRModelError(
                f"Graph requested output {output_index} from a single-output node."
            )
        return value

    def _operation(self, node: Mapping[str, Any], inputs: List[Any], data):
        torch, functional = _torch_modules()
        op = node.get("op")
        name = node.get("name", "")
        attrs = node.get("attrs") or node.get("attr") or {}

        if op == "null":
            if name == "data":
                return data
            try:
                return self.params[name]
            except KeyError as exc:
                raise FinFindRModelError(f"Missing checkpoint input {name!r}.") from exc

        if op == "Convolution":
            if len(inputs) < 2:
                raise FinFindRModelError("Convolution is missing its weights.")
            bias = None if _bool_attr(attrs, "no_bias", True) else inputs[2]
            return functional.conv2d(
                inputs[0], inputs[1], bias=bias,
                stride=_tuple_attr(attrs, "stride", (1, 1)),
                padding=_tuple_attr(attrs, "pad", (0, 0)),
                dilation=_tuple_attr(attrs, "dilate", (1, 1)),
                groups=int(attrs.get("num_group", 1)),
            )

        if op == "Pad":
            widths = _tuple_attr(attrs, "pad_width", (0,) * (inputs[0].ndim * 2))
            if len(widths) != inputs[0].ndim * 2:
                raise FinFindRModelError(f"Invalid MXNet pad_width: {widths!r}.")
            padding = []
            for axis in reversed(range(inputs[0].ndim)):
                padding.extend((widths[2 * axis], widths[2 * axis + 1]))
            mode = {"edge": "replicate", "constant": "constant", "reflect": "reflect"}.get(
                attrs.get("mode", "constant")
            )
            if mode is None:
                raise FinFindRModelError(f"Unsupported MXNet padding mode: {attrs.get('mode')}.")
            value = float(attrs.get("constant_value", 0.0))
            # PyTorch's non-constant modes accept only trailing dimensions.
            # The pinned FinFindR graph pads H/W and specifies zero N/C pads.
            while len(padding) > 4 and padding[-2:] == [0, 0]:
                padding = padding[:-2]
            return functional.pad(inputs[0], tuple(padding), mode=mode, value=value)

        if op == "BatchNorm":
            tensor, gamma, beta, moving_mean, moving_var = inputs[:5]
            if _bool_attr(attrs, "fix_gamma", False):
                gamma = torch.ones_like(gamma)
            return functional.batch_norm(
                tensor, moving_mean, moving_var, gamma, beta,
                training=False, momentum=float(attrs.get("momentum", 0.9)),
                eps=float(attrs.get("eps", 1e-5)),
            )

        if op == "LeakyReLU":
            kind = attrs.get("act_type", "leaky")
            slope = float(attrs.get("slope", 0.25))
            if kind == "elu":
                return functional.elu(inputs[0], alpha=slope)
            if kind == "leaky":
                return functional.leaky_relu(inputs[0], negative_slope=slope)
            if kind == "prelu":
                return functional.prelu(inputs[0], inputs[1])
            raise FinFindRModelError(f"Unsupported LeakyReLU act_type: {kind}.")

        if op == "Activation":
            kind = attrs.get("act_type")
            if kind == "relu":
                return functional.relu(inputs[0])
            if kind == "sigmoid":
                return torch.sigmoid(inputs[0])
            if kind == "tanh":
                return torch.tanh(inputs[0])
            if kind == "softrelu":
                return functional.softplus(inputs[0])
            if kind == "softsign":
                return functional.softsign(inputs[0])
            raise FinFindRModelError(f"Unsupported Activation act_type: {kind}.")

        if op == "L2Normalization":
            mode = attrs.get("mode", "instance")
            eps = float(attrs.get("eps", 1e-10))
            if mode == "spatial":
                axes = tuple(range(2, inputs[0].ndim))
                norm = torch.sqrt(torch.sum(torch.square(inputs[0]), dim=axes, keepdim=True) + eps)
                return inputs[0] / norm
            if mode == "channel":
                return functional.normalize(inputs[0], p=2.0, dim=1, eps=eps)
            if mode == "instance":
                flat = inputs[0].flatten(start_dim=1)
                return functional.normalize(flat, p=2.0, dim=1, eps=eps).reshape_as(inputs[0])
            raise FinFindRModelError(f"Unsupported L2Normalization mode: {mode}.")

        if op == "FullyConnected":
            tensor = inputs[0]
            if _bool_attr(attrs, "flatten", True):
                tensor = tensor.flatten(start_dim=1)
            bias = None if _bool_attr(attrs, "no_bias", False) else inputs[2]
            return functional.linear(tensor, inputs[1], bias)

        if op == "_random_uniform":
            low, high = float(attrs.get("low", 0)), float(attrs.get("high", 1))
            shape = _tuple_attr(attrs, "shape", ())
            if low != high:
                raise FinFindRModelError(
                    "A stochastic MXNet random node is not supported for reproducible inference."
                )
            return torch.full(shape, low, dtype=torch.float32, device=self.device)

        if op == "SliceChannel":
            axis = int(attrs.get("axis", 1))
            count = int(attrs["num_outputs"])
            chunks = torch.chunk(inputs[0], count, dim=axis)
            if _bool_attr(attrs, "squeeze_axis", False):
                chunks = tuple(chunk.squeeze(axis) for chunk in chunks)
            return tuple(chunks)

        if op == "Concat":
            return torch.cat(inputs, dim=int(attrs.get("dim", 1)))
        if op == "tile":
            return torch.tile(inputs[0], _tuple_attr(attrs, "reps", (1,) * inputs[0].ndim))
        if op == "BlockGrad":
            return inputs[0]

        if op in {"elemwise_add", "broadcast_add"}:
            return inputs[0] + inputs[1]
        if op in {"elemwise_mul", "broadcast_mul"}:
            return inputs[0] * inputs[1]
        if op in {"elemwise_div", "broadcast_div"}:
            return inputs[0] / inputs[1]
        if op == "broadcast_lesser":
            return (inputs[0] < inputs[1]).to(dtype=inputs[0].dtype)

        scalar = float(attrs.get("scalar", 0.0))
        if op == "_plus_scalar":
            return inputs[0] + scalar
        if op == "_minus_scalar":
            return inputs[0] - scalar
        if op == "_mul_scalar":
            return inputs[0] * scalar

        unary = {
            "cos": torch.cos,
            "exp": torch.exp,
            "log": torch.log,
            "negative": torch.neg,
            "sin": torch.sin,
            "sqrt": torch.sqrt,
            "square": torch.square,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
            "ones_like": torch.ones_like,
            "zeros_like": torch.zeros_like,
        }
        if op in unary:
            return unary[op](inputs[0])
        if op == "sum":
            axis = _literal(attrs.get("axis"), None)
            if isinstance(axis, list):
                axis = tuple(axis)
            return torch.sum(inputs[0], dim=axis, keepdim=_bool_attr(attrs, "keepdims", False))

        raise FinFindRModelError(f"Unsupported MXNet operator {op!r} at node {name!r}.")
