"""Tests for trajectory direction decomposition."""

import unittest

import torch

from experiments.analyze_path_following import decompose


class DecomposeTest(unittest.TestCase):
    """Verify parallel strength and off-path motion metrics."""

    def test_scaled_parallel_path(self) -> None:
        """A half-strength aligned path has no orthogonal drift."""
        result = decompose(torch.tensor([1.0, 0.0]), torch.tensor([2.0, 0.0]))
        self.assertAlmostEqual(result["cosine"], 1.0)
        self.assertAlmostEqual(result["parallel_strength"], 0.5)
        self.assertAlmostEqual(result["generated_to_real_norm"], 0.5)
        self.assertAlmostEqual(result["orthogonal_to_real_norm"], 0.0)
        self.assertAlmostEqual(result["direction_explained_fraction"], 1.0)

    def test_orthogonal_path(self) -> None:
        """An equally long perpendicular path is entirely unexplained."""
        result = decompose(torch.tensor([0.0, 2.0]), torch.tensor([2.0, 0.0]))
        self.assertAlmostEqual(result["cosine"], 0.0)
        self.assertAlmostEqual(result["parallel_strength"], 0.0)
        self.assertAlmostEqual(result["generated_to_real_norm"], 1.0)
        self.assertAlmostEqual(result["orthogonal_to_real_norm"], 1.0)
        self.assertAlmostEqual(result["direction_explained_fraction"], 0.0)


if __name__ == "__main__":
    unittest.main()
