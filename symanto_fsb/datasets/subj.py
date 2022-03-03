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

"""An English subjectivity dataset."""

import hashlib
from pathlib import Path
from typing import List

import datasets

_CITATION = """\
@inproceedings{subj,
    title = "A Sentimental Education: Sentiment Analysis Using Subjectivity Summarization Based on Minimum Cuts",
    author = "Pang, Bo  and
      Lee, Lillian",
    booktitle = "Proceedings of the 42nd Annual Meeting of the Association for Computational Linguistics ({ACL}-04)",
    month = jul,
    year = "2004",
    address = "Barcelona, Spain",
    url = "https://aclanthology.org/P04-1035",
    doi = "10.3115/1218955.1218990",
    pages = "271--278",
}
"""

_DESCRIPTION = """An English subjectivity dataset."""

_HOMEPAGE = "http://www.cs.cornell.edu/people/pabo/movie-review-data"

_LICENSE = "Unknown."

_URL = (
    "http://www.cs.cornell.edu/people/pabo/movie-review-data/rotten_imdb.tar.gz"
)


def _hash(example_with_id) -> str:
    """Hashes a single example."""
    key = example_with_id[0], example_with_id[1]["text"]
    key_bytes = str(key).encode("utf8")
    return hashlib.sha256(key_bytes).hexdigest()


def _split(examples, sizes: List[int]):
    """Creates a random split of the data."""
    cum_sizes = []
    for i in range(len(sizes)):
        cum_sizes.append(sum(sizes[: i + 1]))
    num_shards = sum(sizes)
    for example in examples:
        h = int(_hash(example), 16) % num_shards
        for index, size in enumerate(cum_sizes):
            if h < size:
                yield example, index
                break


def read(data_path: Path, split: str):
    def get_examples():
        for label, name in [
            ("subjective", "quote.tok.gt9.5000"),
            ("objective", "plot.tok.gt9.5000"),
        ]:
            filepath = data_path.joinpath(name)
            with filepath.open("tr", encoding="Windows-1252") as reader:
                for index, line in enumerate(reader):
                    text = line.strip()
                    if not text:
                        continue
                    yield str(index), {
                        "text": text,
                        "label": label,
                    }

    split_to_index = {"test": 0, "dev": 1, "train": 2}
    for (_id, ex), split_index in _split(get_examples(), [20, 20, 60]):
        if split_index == split_to_index[split]:
            yield f"{split}_{ex['label']}_{int(_id):05}", ex


class SubjDataset(datasets.GeneratorBasedBuilder):
    """An English subjectivity dataset."""

    VERSION = datasets.Version("1.1.0")  # type: ignore

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(  # type: ignore
            name="subj",
            version=VERSION,
            description="An English subjectivity dataset.",
        ),
    ]

    def _info(self):
        features = datasets.Features(
            {
                "text": datasets.Value("string"),
                "label": datasets.ClassLabel(names=["subjective", "objective"]),
            },
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_dir = dl_manager.download_and_extract(_URL)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_dir": data_dir,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "data_dir": data_dir,
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data_dir": data_dir,
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(
        self,
        data_dir,
        split,
    ):
        """Yields examples as (key, example) tuples."""
        yield from read(Path(data_dir), split)
