# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.benchmark.utils.utils import get_number_iterations


class TestIterationAlignment(unittest.TestCase):
    def test_shape_cycles_are_rounded_up(self):
        for iterations, expected in [(1, 3), (2, 3), (4, 6), (5, 6), (7, 9)]:
            with self.subTest(iterations=iterations):
                self.assertEqual(
                    get_number_iterations(iterations, 1, 3, "async"), expected
                )

    def test_complete_shape_cycles_are_preserved(self):
        for iterations in [3, 6, 9]:
            with self.subTest(iterations=iterations):
                self.assertEqual(
                    get_number_iterations(iterations, 2, 3, "async"), iterations
                )

    def test_request_alignment_is_preserved(self):
        for shapes in [1, 2, 4]:
            with self.subTest(shapes=shapes):
                self.assertEqual(get_number_iterations(5, 4, shapes, "async"), 8)

    def test_sync_iterations_are_preserved(self):
        self.assertEqual(get_number_iterations(5, 1, 3, "sync"), 5)

    def test_time_limited_iterations_are_preserved(self):
        for iterations in [0, None]:
            with self.subTest(iterations=iterations):
                self.assertEqual(
                    get_number_iterations(iterations, 1, 3, "async"), iterations
                )


if __name__ == "__main__":
    unittest.main()
