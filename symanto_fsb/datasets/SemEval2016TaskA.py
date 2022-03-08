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

"""Data of SemEval 2016 Task A."""

import csv
from pathlib import Path

import datasets

_CITATION = """\
@InProceedings{semeval2016,
author    = {Preslav Nakov and Alan Ritter and Sara Rosenthal and Veselin Stoyanov and Fabrizio Sebastiani},
title     = {{SemEval}-2016 Task 4: Sentiment Analysis in {T}witter},
booktitle = {Proceedings of the 10th International Workshop on Semantic Evaluation},
series    = {SemEval '16},
month     = {June},
year      = {2016},
address   = {San Diego, California},
publisher = {Association for Computational Linguistics},
}
"""

_DESCRIPTION = """\
English twitter sentiment polarity task (SemEval 2016 Task A).
"""

_HOMEPAGE = "https://alt.qcri.org/semeval2017/task4/?id=download-the-full-training-data-for-semeval-2017-task-4"

_LICENSE = ""

_URL = "https://www.dropbox.com/s/byzr8yoda6bua1b/2017_English_final.zip?dl=1"


class Semeval2016Dataset(datasets.GeneratorBasedBuilder):
    """A twitter sentiment polarity dataset."""

    VERSION = datasets.Version("1.1.0")  # type: ignore

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(  # type: ignore
            name="semeval2016",
            version=VERSION,
            description="Trinary sentiment task on English Twitter data.",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
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
            ),
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
                    "split": datasets.Split.TRAIN,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "data_dir": data_dir,
                    "split": datasets.Split.TEST,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data_dir": data_dir,
                    "split": datasets.Split.VALIDATION,
                },
            ),
        ]

    def _generate_examples(
        self,
        data_dir,
        split,
    ):
        """Yields examples as (key, example) tuples."""
        data_path = Path(data_dir).joinpath(
            "2017_English_final", "GOLD", "Subtask_A"
        )
        filenames = {
            datasets.Split.TRAIN: ["twitter-2016train-A.txt"],
            datasets.Split.VALIDATION: [
                "twitter-2016dev-A.txt",
                "twitter-2016devtest-A.txt",
            ],
            datasets.Split.TEST: ["twitter-2016test-A.txt"],
        }
        for filename in filenames[split]:
            path = data_path.joinpath(filename)
            with path.open("rt") as reader:
                for _id, row in enumerate(
                    csv.reader(
                        reader,
                        delimiter="\t",
                    )
                ):
                    yield f"{_id}_{row[0]}", {"text": row[2], "label": row[1]}
