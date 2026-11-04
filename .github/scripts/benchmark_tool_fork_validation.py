# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Fork-only validation; this script is excluded from the upstream PRs."""

import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'tools/benchmark_tool/openvino/tools/benchmark/utils/inputs_filling.py'


def load_checkout_tools():
    import openvino.tools
    openvino.tools.__path__.insert(0, str(ROOT / 'tools/benchmark_tool/openvino/tools'))


def run_tests(arguments):
    import pytest
    load_checkout_tools()
    return pytest.main(arguments)


def run_benchmark(arguments):
    load_checkout_tools()
    from openvino.tools.benchmark.main import main
    sys.argv = [sys.argv[0], *arguments]
    return main()


def validate():
    import numpy as np
    import openvino as ov

    base = os.environ['BASE_SHA']
    regression = os.environ['REGRESSION_TEST']
    subprocess.run(['git', 'fetch', '--no-tags', '--depth=1', 'origin', base], cwd=ROOT, check=True)
    original = subprocess.check_output(['git', 'show', f'{base}:{SOURCE.relative_to(ROOT).as_posix()}'], cwd=ROOT)
    fixed = SOURCE.read_bytes()
    print('Runtime:', ov.get_version(), 'NumPy:', np.__version__, flush=True)
    print('Fixed source SHA256:', hashlib.sha256(fixed).hexdigest(), flush=True)
    try:
        SOURCE.write_bytes(original)
        baseline = subprocess.run([sys.executable, __file__, 'unit', '-q', regression], cwd=ROOT, timeout=120)
        if baseline.returncode != 1:
            raise RuntimeError(f'Expected regression failures on the baseline, got exit {baseline.returncode}')
    finally:
        SOURCE.write_bytes(fixed)
    subprocess.run(
        [sys.executable, __file__, 'unit', '-q', 'tools/benchmark_tool/unit_tests'],
        cwd=ROOT, timeout=120, check=True,
    )

    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        model_path = directory / 'model.xml'
        if 'named_input' in regression:
            name = 'model/input-0:0'
            parameter = ov.opset13.parameter([1, 4], ov.Type.f32)
            parameter.output(0).get_tensor().set_names({name})
            model = ov.Model([ov.opset13.relu(parameter)], [parameter])
            input_path = directory / 'input.npy'
            np.save(input_path, np.array([[-1, 0, 1, 2]], dtype=np.float32))
            inputs = ['-i', f'{name}:{input_path}']
        else:
            parameter = ov.opset13.parameter([128], ov.Type.boolean)
            model = ov.Model([ov.opset13.logical_not(parameter)], [parameter])
            inputs = []
        ov.serialize(model, str(model_path))
        subprocess.run(
            [sys.executable, __file__, 'benchmark', '-m', str(model_path), '-d', 'CPU',
             '-api', 'sync', '-hint', 'latency', '-nireq', '1', '-niter', '1', *inputs],
            cwd=ROOT, timeout=120, check=True,
        )
    subprocess.run(['git', 'diff', '--exit-code', '--', str(SOURCE.relative_to(ROOT))], cwd=ROOT, check=True)
    print('Baseline reproduction, regression suite, and native CPU benchmark passed.', flush=True)


if __name__ == '__main__':
    command, *arguments = sys.argv[1:]
    if command == 'unit':
        raise SystemExit(run_tests(arguments))
    if command == 'benchmark':
        raise SystemExit(run_benchmark(arguments))
    validate()
