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

"""An English acceptability dataset."""

import csv
from pathlib import Path

import datasets

_CITATION = """\
@article{warstadt-etal-2019-neural,
    title = "Neural Network Acceptability Judgments",
    author = "Warstadt, Alex  and
      Singh, Amanpreet  and
      Bowman, Samuel R.",
    journal = "Transactions of the Association for Computational Linguistics",
    volume = "7",
    month = mar,
    year = "2019",
    url = "https://aclanthology.org/Q19-1040",
    doi = "10.1162/tacl_a_00290",
    pages = "625--641",
    abstract = "This paper investigates the ability of artificial neural networks to judge the grammatical acceptability of a sentence, with the goal of testing their linguistic competence. We introduce the Corpus of Linguistic Acceptability (CoLA), a set of 10,657 English sentences labeled as grammatical or ungrammatical from published linguistics literature. As baselines, we train several recurrent neural network models on acceptability classification, and find that our models outperform unsupervised models by Lau et al. (2016) on CoLA. Error-analysis on specific grammatical phenomena reveals that both Lau et al.{'}s models and ours learn systematic generalizations like subject-verb-object order. However, all models we test perform far below human level on a wide range of grammatical constructions.",
}
"""

_DESCRIPTION = """An English acceptability dataset."""

_HOMEPAGE = "https://nyu-mll.github.io/CoLA"

_LICENSE = "Unknown."

_URL = "https://nyu-mll.github.io/CoLA/cola_public_1.1.zip"


_labels = ["unacceptable", "acceptable"]


def _to_label(index):
    return _labels[index]


def read(data_path: Path, split: str):
    splits_to_filenames = {
        "train": ("in_domain_train",),
        "test": ("in_domain_dev", "out_of_domain_dev"),
    }

    for filename in splits_to_filenames[split]:
        filepath = data_path.joinpath(
            "cola_public", "raw", filename
        ).with_suffix(".tsv")
        with filepath.open("rt") as reader:
            rows = csv.DictReader(
                reader,
                delimiter="\t",
                fieldnames=["id", "label", "*", "text"],
            )
            for index, row in enumerate(rows):
                assert row["label"]
                assert row["id"]
                class_index = int(row["label"])
                label = _to_label(class_index)
                yield f"{filename}_{row['id']}_{index:04}", {
                    "text": row["text"],
                    "label": label,
                }


class ColaDataset(datasets.GeneratorBasedBuilder):
    """An English acceptability dataset."""

    VERSION = datasets.Version("1.1.0")  # type: ignore

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(  # type: ignore
            name="cola",
            version=VERSION,
            description="An English acceptability dataset.",
        ),
    ]

    def _info(self):
        features = datasets.Features(
            {
                "text": datasets.Value("string"),
                "label": datasets.ClassLabel(names=_labels),
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
