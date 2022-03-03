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

"""A German emotion dataset."""

import collections
import csv
import hashlib
from pathlib import Path
from typing import List

import datasets

_CITATION = """\
@inproceedings{deisear,
    title = "Crowdsourcing and Validating Event-focused Emotion Corpora for {G}erman and {E}nglish",
    author = "Troiano, Enrica  and
      Pad{\'o}, Sebastian  and
      Klinger, Roman",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1391",
    doi = "10.18653/v1/P19-1391",
    pages = "4005--4011",
    abstract = "Sentiment analysis has a range of corpora available across multiple languages. For emotion analysis, the situation is more limited, which hinders potential research on crosslingual modeling and the development of predictive models for other languages. In this paper, we fill this gap for German by constructing deISEAR, a corpus designed in analogy to the well-established English ISEAR emotion dataset. Motivated by Scherer{'}s appraisal theory, we implement a crowdsourcing experiment which consists of two steps. In step 1, participants create descriptions of emotional events for a given emotion. In step 2, five annotators assess the emotion expressed by the texts. We show that transferring an emotion classification model from the original English ISEAR to the German crowdsourced deISEAR via machine translation does not, on average, cause a performance drop.",
}
"""

_DESCRIPTION = """A German emotion dataset."""

_HOMEPAGE = (
    "https://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/deisear"
)

_LICENSE = "Unknown."

_URL = "https://www.ims.uni-stuttgart.de/documents/ressourcen/korpora/emotions/deISEARenISEAR.zip"


_labels = {
    "schuld": "guilt",
    "wut": "anger",
    "ekel": "disgust",
    "angst": "fear",
    "freude": "joy",
    "scham": "shame",
    "traurigkeit": "sadness",
}


def _to_emotion_label(values):
    labels = collections.Counter([v["annotation"] for v in values])
    label, count = labels.most_common(n=1)[0]
    if count > len(labels) // 2:
        return _labels[label]
    raise ValueError(f"No majority label: {values}")


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
    filepath = data_path.joinpath("deISEAR_validation.tsv")
    texts = collections.defaultdict(list)
    with filepath.open("rt") as reader:
        for row in csv.DictReader(
            reader,
            delimiter="\t",
        ):
            text = row["Sentence"]
            # fmt: off
            if text.startswith("Ich fühlte ..., "):
                text = text[len("Ich fühlte ..., "):]
            elif text.startswith("ich fühlte ..., "):
                text = text[len("ich fühlte ..., "):]
            elif text.startswith("Ich fühlte ...,,"):
                text = text[len("Ich fühlte ...,,"):]
            elif text.startswith("Ich fühlte ... "):
                text = text[len("Ich fühlte ... "):]
            elif text.startswith("Ich fühle ..., "):
                text = text[len("Ich fühle ..., "):]
            elif text.startswith("Ich fühle ... "):
                text = text[len("Ich fühle ... "):]
            elif text.startswith("Ich fühlte mich ..., "):
                text = text[len("Ich fühlte mich ..., "):]
            # fmt: on
            texts[text].append(
                {
                    "id": row["Sentence_id"],
                    "annotation": row["Annotation"],
                }
            )

    def get_examples():
        for text, values in texts.items():
            try:
                label = _to_emotion_label(values)
            except ValueError:
                continue
            yield values[0]["id"], {
                "text": text,
                "label": label,
            }

    split_to_index = {"test": 0, "dev": 1, "train": 2}
    for ex, split_index in _split(get_examples(), [1, 1, 1]):
        if split_index == split_to_index[split]:
            yield ex


class DeIsearDataset(datasets.GeneratorBasedBuilder):
    """A German emotion dataset."""

    VERSION = datasets.Version("1.1.0")  # type: ignore

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(  # type: ignore
            name="german",
            version=VERSION,
            description="German emotion data.",
        ),
    ]

    def _info(self):
        features = datasets.Features(
            {
                "text": datasets.Value("string"),
                "label": datasets.ClassLabel(
                    names=[
                        "guilt",
                        "anger",
                        "disgust",
                        "fear",
                        "joy",
                        "shame",
                        "sadness",
                    ],
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
        yield from read(Path(data_dir).joinpath("deISEARenISEAR"), split)
