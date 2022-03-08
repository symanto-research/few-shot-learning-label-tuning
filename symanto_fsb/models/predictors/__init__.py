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

from abc import ABC, abstractmethod
from typing import List, Mapping


def batch(elements, batch_size):
    batch = []
    for x in elements:
        batch.append(x)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


class Predictor(ABC):
    @abstractmethod
    def predict(
        self,
        hypotheses: Mapping[str, str],
        examples: List[dict],
    ) -> List[int]:
        """
        Given label hypotheses and examples, predicts the label index.

        Hypotheses: {
            "positive": "This is a great product!"
            "negative": "This is a terrible product!"
        }

        Examples: [
            {
                "text": "The camera of this phone is terrible!"
            },
            {
                "text": "I love this phone!"
            },
        ]

        Correct predictions: [1, 0]

        Args:
            hypotheses: A dict that maps labels to hypotheses.
            examples: A list of dicts with a text field.

        Returns:
            For each example the position of the label in hypotheses.
        """
        ...

    def few_shot(
        self,
        hypotheses: Mapping[str, str],
        examples: List[dict],
    ):
        """
        Hypotheses: {
            "positive": "This is a great product!"
            "negative": "This is a terrible product!"
        }

        Examples: [
            {
                "text": "The camera of this phone is terrible!"
                "label": "negative"
            },
            {
                "text": "I love this phone!"
                "label": "positive"
            },
        ]

        Args:
            hypotheses: A dict that maps labels to hypotheses.
            examples: A list of dicts with a text and label field.
        """
        raise NotImplementedError()
