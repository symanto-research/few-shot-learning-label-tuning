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

import importlib
from typing import List, Mapping

import numpy as np
from tqdm import tqdm

from symanto_fsb.models.predictors import Predictor, batch
from symanto_fsb.models.siamese_models import Model


class DualEncoderPredictor(Predictor):
    def __init__(self, model_config):
        module_name = model_config["module"]
        module = importlib.import_module(
            f"symanto_fsb.models.siamese_models.{module_name}"
        )
        self.model = module.load_model(model_config)  # type: Model
        self.batch_size = model_config.get("batch_size", 32)

    def score(
        self,
        hypotheses: Mapping[str, str],
        examples: List[dict],
    ) -> List[int]:
        hypothesis_encodings = self.model.encode_batch(
            list(hypotheses.values())
        )
        assert len(hypotheses) == len(hypothesis_encodings)
        matrixes = []
        for batched_examples in tqdm(
            batch(examples, batch_size=self.batch_size),
            total=len(examples) // self.batch_size,
        ):
            text = [e["text"] for e in batched_examples]
            example_encodings = self.model.encode_batch(text)
            matrix = np.inner(example_encodings, hypothesis_encodings)
            matrixes.append(matrix)
        return np.concatenate(matrixes)

    def predict(
        self,
        hypotheses: Mapping[str, str],
        examples: List[dict],
    ) -> List[int]:
        matrix = self.score(hypotheses, examples)
        indexes = np.argmax(matrix, axis=1)
        p_labels = indexes.tolist()
        assert len(p_labels) == len(examples)
        return p_labels

    def num_parameters(self) -> int:
        return self.model.num_parameters()

    def few_shot(
        self,
        hypotheses: Mapping[str, str],
        examples: List[dict],
    ):
        self.model.few_shot(
            hypotheses,
            examples,
            self.batch_size,
        )


def build_predictor(model_config) -> Predictor:
    return DualEncoderPredictor(model_config)
