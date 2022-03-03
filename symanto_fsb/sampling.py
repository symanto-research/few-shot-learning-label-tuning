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

import collections
import hashlib
from typing import Dict, List, Union

import datasets


def _hash(text: str, seed: int) -> str:
    """Hashes a single example."""
    key = str((text, seed)).encode("utf8")
    return hashlib.sha256(key).hexdigest()


def sample(
    dataset: datasets.Dataset, seed: int, n_examples_per_label: int
) -> Dict[str, List[Union[str, int]]]:

    examples_by_label = collections.defaultdict(list)

    hash_to_index = collections.defaultdict(list)

    for idx, row in enumerate(dataset):
        fingerprint = _hash(row["text"], seed)
        examples_by_label[row["label"]].append(fingerprint)
        hash_to_index[fingerprint].append(idx)

    indexes = []

    for examples in examples_by_label.values():
        examples.sort()
        for fingerprint in examples[:n_examples_per_label]:
            indexes.extend(hash_to_index[fingerprint])

    def filter_fn(example, idx) -> bool:
        del example
        return idx in indexes

    return dataset[indexes]  # type: ignore
