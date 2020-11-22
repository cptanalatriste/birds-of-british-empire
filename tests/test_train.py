from unittest import TestCase

import torch
from torch import Tensor

from attnganw.train import get_noise_interpolation


class Test(TestCase):

    def test_get_noise_interpolation(self):
        batch_size = 1
        noise_vector_size = 3

        noise_vector_start: Tensor = torch.randn(batch_size, noise_vector_size, dtype=torch.float)
        noise_vector_end: Tensor = torch.randn(batch_size, noise_vector_size, dtype=torch.float)

        vectors_for_interpolation = get_noise_interpolation(batch_size=batch_size, noise_vector_size=noise_vector_size,
                                                            noise_vector_start=noise_vector_start,
                                                            noise_vector_end=noise_vector_end,
                                                            gpu_id=-1, number_of_steps=2)

        self.assertEqual(len(vectors_for_interpolation), 3)
        self.assertTrue(torch.equal(vectors_for_interpolation[0], noise_vector_start))
        self.assertTrue(torch.equal(vectors_for_interpolation[2], noise_vector_end))
