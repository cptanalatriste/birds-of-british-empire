from typing import List
from unittest import TestCase

import torch
from torch import Tensor

from attnganw import config
from attnganw.randomutils import get_vector_interpolation


class TestInterpolation(TestCase):

    def test_get_noise_interpolation(self):
        batch_size = 1
        noise_vector_size = 3

        noise_vector_start: Tensor = torch.randn(batch_size, noise_vector_size, dtype=torch.float)
        noise_vector_end: Tensor = torch.randn(batch_size, noise_vector_size, dtype=torch.float)

        config.generation['noise_interpolation_steps'] = 4
        initial_interpolation: List[Tensor] = get_vector_interpolation(batch_size=batch_size,
                                                                       noise_vector_size=noise_vector_size,
                                                                       noise_vector_start=noise_vector_start,
                                                                       noise_vector_end=noise_vector_end,
                                                                       gpu_id=-1)

        self.assertEqual(len(initial_interpolation), config.generation['noise_interpolation_steps'] + 1)
        self.assertTrue(torch.equal(initial_interpolation[0], noise_vector_start))
        self.assertTrue(torch.equal(initial_interpolation[-1], noise_vector_end))

        config.generation['noise_interpolation_steps'] = config.generation['noise_interpolation_steps'] * 2
        second_interpolation: List[Tensor] = get_vector_interpolation(batch_size=batch_size,
                                                                      noise_vector_size=noise_vector_size,
                                                                      noise_vector_start=noise_vector_start,
                                                                      noise_vector_end=noise_vector_end,
                                                                      gpu_id=-1)

        self.assertEqual(len(second_interpolation), config.generation['noise_interpolation_steps'] + 1)
        self.assertTrue(torch.equal(second_interpolation[0], noise_vector_start))
        self.assertTrue(torch.equal(second_interpolation[-1], noise_vector_end))
        self.assertFalse(torch.equal(second_interpolation[1], initial_interpolation[1]))
