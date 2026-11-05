# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Validate the benchmark change on the fork without adding upstream CI files."""

import hashlib
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(os.environ.get("OV_BENCHMARK_CHECKOUT", Path(__file__).resolve().parents[2])).resolve()
SOURCE = ROOT / "tools/benchmark_tool/openvino/tools/benchmark/utils/utils.py"
REGRESSION = "tools/benchmark_tool/unit_tests/test_iteration_alignment.py"


def load_tools():
    import openvino.tools

    openvino.tools.__path__.insert(0, str(ROOT / "tools/benchmark_tool/openvino/tools"))


def native_benchmark(model, iterations, expected):
    result = subprocess.run(
        [sys.executable, __file__, "benchmark", "-m", str(model), "-d", "CPU",
         "-api", "async", "-hint", "latency", "-nireq", "1", "-niter", str(iterations),
         "-data_shape", "input[1,2][1,3][1,4]", "-layout", "input[NC]",
         "-inference_only", "false"],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120,
    )
    print(result.stdout, flush=True)
    result.check_returncode()
    count = re.search(r"Count:\s+(\d+) iterations", result.stdout)
    if count is None or int(count.group(1)) != expected:
        raise AssertionError(f"Requested {iterations} iterations: expected count {expected}")


def validate():
    import numpy as np
    import openvino as ov

    base = os.environ["BASE_SHA"]
    if subprocess.run(["git", "cat-file", "-e", base], cwd=ROOT).returncode:
        subprocess.run(["git", "fetch", "--no-tags", "--depth=1", "origin", base], cwd=ROOT, check=True)
    original = subprocess.check_output(["git", "show", f"{base}:{SOURCE.relative_to(ROOT).as_posix()}"], cwd=ROOT)
    fixed = SOURCE.read_bytes()
    print("Runtime:", ov.get_version(), "NumPy:", np.__version__, flush=True)
    print("Fixed source SHA256:", hashlib.sha256(fixed).hexdigest(), flush=True)
    with tempfile.TemporaryDirectory() as temporary:
        parameter = ov.opset13.parameter([1, -1], ov.Type.f32)
        parameter.output(0).get_tensor().set_names({"input"})
        model = ov.Model([ov.opset13.relu(parameter)], [parameter])
        model_path = Path(temporary) / "model.xml"
        ov.serialize(model, str(model_path))
        try:
            SOURCE.write_bytes(original)
            baseline = subprocess.run([sys.executable, __file__, "unit", "-q", REGRESSION], cwd=ROOT, timeout=120)
            if baseline.returncode != 1:
                raise AssertionError(f"Expected baseline regression failures, got {baseline.returncode}")
            native_benchmark(model_path, 3, 6)
            native_benchmark(model_path, 5, 9)
        finally:
            SOURCE.write_bytes(fixed)
        subprocess.run(
            [sys.executable, __file__, "unit", "-q", "tools/benchmark_tool/unit_tests"],
            cwd=ROOT, check=True, timeout=120,
        )
        native_benchmark(model_path, 3, 3)
        native_benchmark(model_path, 5, 6)
    if SOURCE.read_bytes() != fixed:
        raise AssertionError("Validation changed the submitted source")
    print("Baseline reproduced; full benchmark unit suite and native CPU iteration counts passed.", flush=True)


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
