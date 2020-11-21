from typing import Tuple

import torch.nn as nn

from model import RNN_ENCODER, G_DCGAN, G_NET
import torch
from torch import Tensor
import logging


class TextEncoderWrapper:

    def __init__(self, vocabulary_size: int, text_embedding_size: int, state_dict_location: str):
        self.text_encoder: nn.Module = RNN_ENCODER(vocabulary_size, nhidden=text_embedding_size)

        state_dictionary = torch.load(state_dict_location, map_location=lambda storage, loc: storage)
        self.text_encoder.load_state_dict(state_dict=state_dictionary)

        print('Load text encoder from:', state_dict_location)

    def start_evaluation_mode(self, gpu_id: int):
        if gpu_id >= 0:
            self.text_encoder = self.text_encoder.cuda()

        self.text_encoder.eval()

    def extract_semantic_vectors(self, text_descriptions: Tensor, description_sizes: Tensor) -> Tuple[Tensor, Tensor]:
        logging.debug("text_descriptions " + str(text_descriptions) + " description_sizes " + str(description_sizes))

        batch_size: int = text_descriptions.shape[0]

        hidden: Tensor = self.text_encoder.init_hidden(batch_size)
        word_features, sentence_features = self.text_encoder(captions=text_descriptions,
                                                             cap_lens=description_sizes,
                                                             hidden=hidden)

        logging.debug(
            "word_features.shape " + str(word_features.shape) + " sentence_features.shape " + str(
                sentence_features.shape))
        return word_features, sentence_features


class GenerativeNetworkWrapper:

    def __init__(self, is_dc_gan: bool, state_dict_location: str):
        self.generative_network: nn.Module

        if is_dc_gan:
            self.generative_network = G_DCGAN()
        else:
            self.generative_network = G_NET()

        state_dictionary = torch.load(state_dict_location, map_location=lambda storage, loc: storage)
        self.generative_network.load_state_dict(state_dict=state_dictionary)

        print('Load Generative Network from: ', state_dict_location)

    def start_evaluation_mode(self, gpu_id: int):
        if gpu_id >= 0:
            self.generative_network = self.generative_network.cuda()

        self.generative_network.eval()

    def generate_images(self, noise_vector: Tensor, word_features: Tensor, sentence_features: Tensor,
                        mask) -> Tuple[Tensor, Tensor]:

        noise_vector.data.normal_(mean=0, std=1)

        logging.debug("noise_vector.shape " + str(noise_vector.shape))
        generated_images, attention_maps, _, _ = self.generative_network(z_code=noise_vector,
                                                                         sent_emb=sentence_features,
                                                                         word_embs=word_features,
                                                                         mask=mask)

        return generated_images, attention_maps
