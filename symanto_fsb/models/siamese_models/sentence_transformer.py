# Copyright 2022 The Symanto Research Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Mapping, Text

import numpy as np
from sentence_transformers import SentenceTransformer

from symanto_fsb.models.siamese_models import (
    Model,
)  # pylint: disable=import-error


def get_example_labels(
    hypotheses: Mapping[str, str],
    examples: List[dict],
) -> np.ndarray:
    label_to_index = {label: index for index, label in enumerate(hypotheses)}
    example_labels = []
    for example in examples:
        if "scores" in example:
            scores = example["scores"]
            assert scores.shape == (len(label_to_index),)
        else:
            index = label_to_index[example["label"]]
            scores = np.zeros(shape=(len(label_to_index),))
            scores[index] = 1
        example_labels.append(scores)
    example_labels = np.array(example_labels)
    return example_labels


class SentenceTransformerModel(Model):
    def __init__(self, model_config: dict):
        self.model = SentenceTransformer(model_config["model"])
        self.precomputed_embeddings: Dict[str, np.ndarray] = {}

    def _encode_batch(self, text: List[Text]) -> np.ndarray:
        encodings = self.model.encode(text)
        return encodings

    def encode_batch(self, text: List[Text]) -> np.ndarray:
        encodings = self._encode_batch(text)
        if self.precomputed_embeddings is not None:
            encodings = [
                self.precomputed_embeddings[text]
                if text in self.precomputed_embeddings
                else encoding
                for text, encoding in zip(text, encodings)
            ]
        return np.array(encodings)

    def few_shot(
        self,
        hypotheses: Mapping[str, str],
        examples: List[dict],
        batch_size: int,
    ) -> None:
        from symanto_fsb.models.siamese_models import label_tuning

        del batch_size
        label_embeddings = self.encode_batch(list(hypotheses.values()))
        example_embeddings = self.encode_batch([e["text"] for e in examples])
        example_labels = get_example_labels(hypotheses, examples)
        hparams = label_tuning.find_hparams(
            example_embeddings, example_labels, label_embeddings, num_folds=4
        )["hparams"]
        new_label_embeddings = label_tuning.label_tuning(
            example_embeddings, example_labels, label_embeddings, **hparams
        )
        for text, label_embedding in zip(
            hypotheses.values(), new_label_embeddings
        ):
            self.precomputed_embeddings[text] = label_embedding


def load_model(model_config):
    return SentenceTransformerModel(model_config)
