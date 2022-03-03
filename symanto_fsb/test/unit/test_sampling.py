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

import unittest
from collections import Counter

from symanto_fsb.datasets import load_dataset
from symanto_fsb.sampling import sample


class TestSampling(unittest.TestCase):
    def test_sampling(self):
        n_trials = 5
        train = load_dataset("gnad10")["train"]
        for n_examples_per_label in [2, 8, 16]:
            texts_per_trial = []
            for i in range(n_trials):
                try:
                    train_sample = sample(
                        train, seed=i, n_examples_per_label=n_examples_per_label
                    )
                except KeyError as k_error:
                    raise ValueError(i) from k_error
                labels = Counter(train_sample["label"])
                self.assertEqual(
                    len(labels), train.features["label"].num_classes
                )
                for label, count in labels.most_common():
                    self.assertEqual(count, n_examples_per_label)
                texts_per_trial.append(train_sample["text"])

            for i in range(1, len(texts_per_trial)):
                self.assertNotEqual(texts_per_trial[0], texts_per_trial[i])

            for i in range(5):
                train_sample = sample(
                    train, seed=i, n_examples_per_label=n_examples_per_label
                )
                self.assertEqual(texts_per_trial[i], train_sample["text"])


if __name__ == "__main__":
    unittest.main()
