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

from typing import List, Mapping

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from symanto_fsb.models.predictors import Predictor


def to_arrays(examples: List[dict]):
    x = []
    y = []
    for example in examples:
        x.append(example["text"])
        y.append(example["label"])
    return np.array(x), np.array(y)


class CharSvmPredictor(Predictor):
    def __init__(self):
        self.pipeline = None

    def predict(
        self,
        hypotheses: Mapping[str, str],
        examples: List[dict],
    ) -> List[int]:
        del hypotheses
        if self.pipeline is None:
            raise ValueError("Attempting to use an untrained model.")
        x, _ = to_arrays(examples)
        y = self.pipeline.predict(x)
        return [self.label_to_index[label] for label in y]

    def few_shot(
        self,
        hypotheses: Mapping[str, str],
        examples: List[dict],
    ):
        self.pipeline = Pipeline(
            [
                (
                    "charngrams",
                    TfidfVectorizer(analyzer="char", ngram_range=(2, 5)),
                ),
                ("clf", LinearSVC()),
            ]
        )
        train_x, train_y = to_arrays(examples)
        self.pipeline.fit(train_x, train_y)
        self.label_to_index = {l: i for i, l in enumerate(hypotheses.keys())}


def build_predictor(model_config) -> Predictor:
    del model_config
    return CharSvmPredictor()
