# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Fork-only image and native CPU validation; excluded from the upstream PR."""

import hashlib
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(os.environ.get("OV_BENCHMARK_CHECKOUT", Path(__file__).resolve().parents[2])).resolve()
SOURCE = ROOT / "tools/benchmark_tool/openvino/tools/benchmark/utils/inputs_filling.py"
REGRESSION = "tools/benchmark_tool/unit_tests/test_image_resize.py"


def load_tools():
    import openvino.tools

    openvino.tools.__path__.insert(0, str(ROOT / "tools/benchmark_tool/openvino/tools"))


def native_benchmark(model, image, layout, baseline):
    result = subprocess.run(
        [sys.executable, __file__, "benchmark", "-m", str(model), "-i", str(image),
         "-d", "CPU", "-api", "sync", "-hint", "latency", "-nireq", "1", "-niter", "1",
         "-layout", f"input[{layout}]"],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120,
    )
    print(result.stdout, flush=True)
    if baseline:
        image_shape = "(3, 4, 2)" if layout == "NCHW" else "(4, 2, 3)"
        if result.returncode != 1 or f"Image shape {image_shape} is not compatible" not in result.stdout:
            raise AssertionError("Baseline did not reproduce the reversed-dimensions image error")
    else:
        result.check_returncode()
        if re.search(r"Count:\s+1 iterations", result.stdout) is None:
            raise AssertionError("Fixed native CPU benchmark did not complete one iteration")


def validate():
    import cv2
    import numpy as np
    import openvino as ov

    base = os.environ["BASE_SHA"]
    if subprocess.run(["git", "cat-file", "-e", base], cwd=ROOT, stderr=subprocess.DEVNULL).returncode:
        subprocess.run(["git", "fetch", "--no-tags", "--depth=1", "origin", base], cwd=ROOT, check=True)
    original = subprocess.check_output(["git", "show", f"{base}:{SOURCE.relative_to(ROOT).as_posix()}"], cwd=ROOT)
    fixed = SOURCE.read_bytes()
    print("Runtime:", ov.get_version(), "NumPy:", np.__version__, "OpenCV:", cv2.__version__, flush=True)
    print("Fixed source SHA256:", hashlib.sha256(fixed).hexdigest(), flush=True)
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        image = np.arange(24, dtype=np.uint8).reshape(4, 2, 3)
        image_path = directory / "input.png"
        if not cv2.imwrite(str(image_path), image):
            raise AssertionError("Could not create the native regression image")
        cases = []
        for layout, shape in [("NCHW", [1, 3, 2, 4]), ("NHWC", [1, 2, 4, 3])]:
            parameter = ov.opset13.parameter(shape, ov.Type.f32)
            parameter.output(0).get_tensor().set_names({"input"})
            model_path = directory / f"{layout}.xml"
            ov.serialize(ov.Model([ov.opset13.relu(parameter)], [parameter]), str(model_path))
            cases.append((layout, shape, model_path))
        try:
            SOURCE.write_bytes(original)
            baseline = subprocess.run([sys.executable, __file__, "unit", "-q", REGRESSION], cwd=ROOT, timeout=120)
            if baseline.returncode != 1:
                raise AssertionError(f"Expected baseline regression failures, got {baseline.returncode}")
            for layout, _, model_path in cases:
                native_benchmark(model_path, image_path, layout, baseline=True)
        finally:
            SOURCE.write_bytes(fixed)
        subprocess.run([sys.executable, __file__, "unit", "-q", "tools/benchmark_tool/unit_tests"], cwd=ROOT, check=True, timeout=120)
        load_tools()
        from openvino.tools.benchmark.utils.inputs_filling import get_image_tensors
        from openvino.tools.benchmark.utils.utils import AppInputInfo

        resized = cv2.resize(image, (4, 2))
        for layout, shape, model_path in cases:
            info = AppInputInfo()
            info.name, info.element_type = "input", ov.Type.u8
            info.layout, info.partial_shape = ov.Layout(layout), ov.PartialShape(shape)
            tensor = get_image_tensors([str(image_path)], info, [1])[0]
            expected = resized.transpose(2, 0, 1) if layout == "NCHW" else resized
            np.testing.assert_array_equal(tensor.data[0], expected)
            native_benchmark(model_path, image_path, layout, baseline=False)
    if SOURCE.read_bytes() != fixed:
        raise AssertionError("Validation changed the submitted source")
    print("Baseline reproduced; benchmark unit suite, real image pixels, and NCHW/NHWC native CPU checks passed.", flush=True)


if __name__ == "__main__":
    mode, *arguments = sys.argv[1:]
    if mode == "unit":
        import pytest

        load_tools()
        raise SystemExit(pytest.main(arguments))
    if mode == "benchmark":
        import logging

        logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
        load_tools()
        from openvino.tools.benchmark.main import main

        sys.argv = [sys.argv[0], *arguments]
        raise SystemExit(main())
    validate()
