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

"""A German sentiment dataset based on Twitter."""

import csv
from pathlib import Path

import datasets

_CITATION = """\
@inproceedings{sb10k,
    title = "A {T}witter Corpus and Benchmark Resources for {G}erman Sentiment Analysis",
    author = "Cieliebak, Mark  and
      Deriu, Jan Milan  and
      Egger, Dominic  and
      Uzdilli, Fatih",
    booktitle = "Proceedings of the Fifth International Workshop on Natural Language Processing for Social Media",
    month = apr,
    year = "2017",
    address = "Valencia, Spain",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/W17-1106",
    doi = "10.18653/v1/W17-1106",
    pages = "45--51",
    abstract = "In this paper we present SB10k, a new corpus for sentiment analysis with approx. 10,000 German tweets. We use this new corpus and two existing corpora to provide state-of-the-art benchmarks for sentiment analysis in German: we implemented a CNN (based on the winning system of SemEval-2016) and a feature-based SVM and compare their performance on all three corpora. For the CNN, we also created German word embeddings trained on 300M tweets. These word embeddings were then optimized for sentiment analysis using distant-supervised learning. The new corpus, the German word embeddings (plain and optimized), and source code to re-run the benchmarks are publicly available.",
}
"""

_DESCRIPTION = """\
A German sentiment dataset based on Twitter.
"""

_HOMEPAGE = "https://spinningbytes.com/more/resources/"

_LICENSE = "Unknown"

_URL = "https://drive.google.com/uc?export=download&id=1wM72cCkm0D2RNuH1tAWZKAwRoLjUrE3a"


def read(data_path: Path, split: str):
    filepath = data_path.joinpath("sb10k", f"de_{split}.tsv")
    with filepath.open("rt") as reader:
        for row in csv.reader(
            reader,
            delimiter="\t",
        ):
            yield row[0], {"text": row[3], "label": row[2]}


class Sb10kDataset(datasets.GeneratorBasedBuilder):
    """A German sentiment dataset based on Twitter."""

    VERSION = datasets.Version("1.1.0")  # type: ignore

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(  # type: ignore
            name="sentiment",
            version=VERSION,
            description="Sentiment polarity labels.",
        ),
    ]

    def _info(self):
        features = datasets.Features(
            {
                "text": datasets.Value("string"),
                "label": datasets.ClassLabel(
                    names=[
                        "positive",
                        "negative",
                        "neutral",
                    ]
                ),
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
        ]

    def _generate_examples(
        self,
        data_dir,
        split,
    ):
        """Yields examples as (key, example) tuples."""
        yield from read(Path(data_dir), split)
