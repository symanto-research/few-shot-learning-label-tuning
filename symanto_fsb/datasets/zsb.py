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

"""A benchmark of three dataset for Zero-Shot text classification."""

import csv
from pathlib import Path

import datasets

_CITATION = """\
@inproceedings{zsb,
    title={Benchmarking Zero-shot Text Classification: Datasets, Evaluation and Entailment Approach},
    author={Wenpeng Yin, Jamaal Hay and Dan Roth},
    booktitle={{EMNLP}},
    url = {https://arxiv.org/abs/1909.00161},
    year={2019}
}

@inproceedings{topic,
 author = {Zhang, Xiang and Zhao, Junbo and LeCun, Yann},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {C. Cortes and N. Lawrence and D. Lee and M. Sugiyama and R. Garnett},
 pages = {},
 publisher = {Curran Associates, Inc.},
 title = {Character-level Convolutional Networks for Text Classification},
 url = {https://proceedings.neurips.cc/paper/2015/file/250cf8b51c773f3f8dc8b4be867a9a02-Paper.pdf},
 volume = {28},
 year = {2015}
}

@inproceedings{situation,
  title={University of Illinois LoReHLT 17 Submission},
  author={Stephen Mayhew and Chase Duncan and M. Sammons and Chen-Tse Tsai and X. Li and Haojie Pan and Sheng Zhou and Jennifer Zou and Y. Song},
  year={2017}
}

@inproceedings{emotions,
  author = {Bostan, Laura Ana Maria and Klinger, Roman},
  title = {An Analysis of Annotated Corpora for Emotion Classification in Text},
  booktitle = {Proceedings of the 27th International Conference on Computational Linguistics},
  year = {2018},
  publisher = {Association for Computational Linguistics},
  pages = {2104--2119},
  location = {Santa Fe, New Mexico, USA},
  url = {http://aclweb.org/anthology/C18-1179},
  pdf = {http://aclweb.org/anthology/C18-1179.pdf}
}
"""

_DESCRIPTION = """\
A benchmark of three datasets for Zero-Shot text classification.
"""

_HOMEPAGE = "https://github.com/yinwenpeng/BenchmarkingZeroShot"

_LICENSE = "The different datasets have different licenses."

_URL = "https://drive.google.com/uc?export=download&id=1qGmyEVD19ruvLLz9J0QGV7rsZPFEz2Az"


def read(
    data_path: Path, split: str, split_label: bool, fieldnames=["label", "text"]
):
    label_names = None
    if data_path.joinpath("classes.txt").exists():
        label_names = []
        with data_path.joinpath("classes.txt").open("tr") as reader:
            for line in reader:
                label = line.strip()
                if label:
                    label_names.append(label)
    filenames = {
        "train": ("train_pu_half_v0", "train_pu_half_v1"),
        "dev": ("dev",),
        "test": ("test",),
    }
    for filename in filenames[split]:
        filepath = data_path.joinpath(data_path, filename).with_suffix(".txt")
        with filepath.open("rt") as reader:
            csv_reader = csv.DictReader(
                reader,
                delimiter="\t",
                fieldnames=fieldnames,
            )
            for index, row in enumerate(csv_reader):
                label = row["label"]
                labels = label.split() if split_label else [label]
                if label_names:
                    labels = [label_names[int(label)] for label in labels]
                yield f"{filename}_{index:04}", {
                    "text": row["text"],
                    "label": labels if split_label else labels[0],
                }


class BenchmarkingZeroShotDataDataset(datasets.GeneratorBasedBuilder):
    """A meta-dataset for few-shot text classification."""

    VERSION = datasets.Version("1.1.0")  # type: ignore

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(  # type: ignore
            name="topic",
            version=VERSION,
            description="Topic classifcation dataset based on Yahoo news groups.",
        ),
        datasets.BuilderConfig(  # type: ignore
            name="emotion",
            version=VERSION,
            description="An emotion detection dataset (based on Unified emotions).",
        ),
        datasets.BuilderConfig(  # type: ignore
            name="situation",
            version=VERSION,
            description="An emergency situation dataset.",
        ),
    ]

    def _info(self):
        if self.config.name == "topic":
            features = datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "label": datasets.ClassLabel(
                        names=[
                            "Society & Culture",
                            "Science & Mathematics",
                            "Health",
                            "Education & Reference",
                            "Computers & Internet",
                            "Sports",
                            "Business & Finance",
                            "Entertainment & Music",
                            "Family & Relationships",
                            "Politics & Government",
                        ]
                    ),
                },
            )
        elif self.config.name == "emotion":
            features = datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "label": datasets.ClassLabel(
                        names=[
                            "anger",
                            "disgust",
                            "fear",
                            "guilt",
                            "joy",
                            "love",
                            "noemo",
                            "sadness",
                            "shame",
                            "surprise",
                        ]
                    ),
                },
            )
        elif self.config.name == "situation":
            features = datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "label": datasets.Sequence(datasets.Value("string")),
                },
            )
        else:
            raise NotImplementedError(self.config.name)
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
        data_path = Path(data_dir).joinpath(
            "BenchmarkingZeroShot", self.config.name
        )
        if self.config.name == "topic":
            yield from read(data_path, split, split_label=False)
        elif self.config.name == "situation":
            yield from read(data_path, split, split_label=True)
        elif self.config.name == "emotion":
            yield from read(
                data_path,
                split,
                split_label=False,
                fieldnames=["label", "tag", "text"],
            )
        else:
            raise NotImplementedError(self.config.name)
