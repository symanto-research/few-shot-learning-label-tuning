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

"""A Spanish emotion and sentiment dataset based on Twitter."""

import collections
import hashlib
from pathlib import Path
from typing import DefaultDict, Dict, List

import datasets
import rdflib

_CITATION = """\
@InProceedings{10.1007/978-3-319-66429-3_68,
    author="Navas-Loro, Mar{\'i}a
    and Rodr{\'i}guez-Doncel, V{\'i}ctor
    and Santana-Perez, Idafen
    and S{\'a}nchez, Alberto",
    editor="Karpov, Alexey
    and Potapova, Rodmonga
    and Mporas, Iosif",
    title="Spanish Corpus for Sentiment Analysis Towards Brands",
    booktitle="Speech and Computer",
    year="2017",
    publisher="Springer International Publishing",
    address="Cham",
    pages="680--689",
    isbn="978-3-319-66429-3"
}
"""

_DESCRIPTION = """\
A Spanish emotion and sentiment dataset based on Twitter.
"""

_HOMEPAGE = "http://sabcorpus.linkeddata.es"

_LICENSE = "Annotations under CC-BY 4.0 for the text see the homepage."

_URL = "http://sabcorpus.linkeddata.es/corpus.zip"


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


_properties = {
    "id": "http://rdfs.org/sioc/spec/id",
    "content": "http://rdfs.org/sioc/spec/content",
    "emotion": "http://gsi.dit.upm.es/ontologies/onyx/hasEmotion",
    "sentiment": "http://gsi.dit.upm.es/ontologies/marl/hasPolarity",
    "brand": "http://gsi.dit.upm.es/ontologies/marl/describesObject",
}

_labels = {
    "http://gsi.dit.upm.es/ontologies/marl/positive": "positive",
    "http://gsi.dit.upm.es/ontologies/marl/negative": "negative",
    "neutral": "neutral",
    "http://sabcorpus.linkeddata.es/vocab#hate": "hate",
    "http://sabcorpus.linkeddata.es/vocab#odio": "hate",
    "http://sabcorpus.linkeddata.es/vocab#dissatisfaction": "dissatisfaction",
    "http://sabcorpus.linkeddata.es/vocab#insatisfaccion": "dissatisfaction",
    "http://sabcorpus.linkeddata.es/vocab#love": "love",
    "http://sabcorpus.linkeddata.es/vocab#amor": "love",
    "http://sabcorpus.linkeddata.es/vocab#satisfaction": "satisfaction",
    "http://sabcorpus.linkeddata.es/vocab#satisfaccion": "satisfaction",
    "http://sabcorpus.linkeddata.es/vocab#happiness": "happiness",
    "http://sabcorpus.linkeddata.es/vocab#felicidad": "happiness",
    "http://sabcorpus.linkeddata.es/vocab#sadness": "sadness",
    "http://sabcorpus.linkeddata.es/vocab#tristeza": "sadness",
    "http://sabcorpus.linkeddata.es/vocab#fear": "fear",
    "http://sabcorpus.linkeddata.es/vocab#temor": "fear",
    "http://sabcorpus.linkeddata.es/vocab#trust": "trust",
    "http://sabcorpus.linkeddata.es/vocab#confianza": "trust",
    "http://sabcorpus.linkeddata.es/vocab#nc2": "no_emotion",
}


def _to_label(v):
    return _labels[v]


def _get_first(elements):
    if len(elements) != 1:
        raise ValueError(elements)
    return elements[0]


def read(data_path: Path, split: str, task: str):
    filepath = data_path.joinpath("corpus.n3")
    g = rdflib.Graph()
    g.parse(str(filepath), format="n3")
    data: DefaultDict[str, Dict[str, List[str]]] = collections.defaultdict(dict)
    for subj, predicate, obj in g:
        for propert_name, property_id in _properties.items():
            if str(predicate) == property_id:
                if propert_name not in data[subj]:
                    data[subj][propert_name] = []
                data[subj][propert_name].append(str(obj))
                break

    def get_examples():
        for datum in data.values():
            if "sentiment" not in datum:
                datum["sentiment"] = ["neutral"]
            if task == "sentiment":
                labels = [_to_label(v) for v in datum["sentiment"]]
            elif task == "emotion":
                labels = [_to_label(v) for v in datum["emotion"]]
            else:
                raise NotImplementedError(labels)
            if len(labels) != 1:
                continue
            yield _get_first(datum["id"]), {
                "text": _get_first(datum["content"]),
                "label": labels[0],
            }

    examples_with_split = _split(get_examples(), [10, 10, 80])
    split_to_index = {"test": 0, "dev": 1, "train": 2}
    for ((_id, ex), split_index) in examples_with_split:
        if split_index == split_to_index[split]:
            yield _id, ex


class SabDataset(datasets.GeneratorBasedBuilder):
    """A Spanish emotion and sentiment dataset based on Twitter."""

    VERSION = datasets.Version("1.1.0")  # type: ignore

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(  # type: ignore
            name="sentiment",
            version=VERSION,
            description="Sentiment polarity labels.",
        ),
        datasets.BuilderConfig(  # type: ignore
            name="emotion",
            version=VERSION,
            description="Emotion labels.",
        ),
    ]

    DEFAULT_CONFIG_NAME = "sentiment"  # type: ignore

    def _info(self):
        features = datasets.Features(
            {
                "text": datasets.Value("string"),
                "label": (
                    datasets.ClassLabel(
                        names=[
                            "positive",
                            "neutral",
                            "negative",
                        ]
                    )
                    if self.config.name == "sentiment"
                    else datasets.ClassLabel(
                        names=[
                            "hate",
                            "dissatisfaction",
                            "love",
                            "satisfaction",
                            "happiness",
                            "sadness",
                            "fear",
                            "trust",
                            "no_emotion",
                        ]
                    )
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
        yield from read(Path(data_dir), split, self.config.name)
