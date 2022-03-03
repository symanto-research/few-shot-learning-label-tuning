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

"""A meta-dataset for evaluating few-shot learning models for text classification."""

from typing import List, Mapping, Optional, Tuple, no_type_check

import datasets

from symanto_fsb.datasets import (
    SemEval2016TaskA,
    cola,
    deisear,
    sab,
    sb10k,
    subj,
    zsb,
)

_modules = {
    "SemEval2016TaskA": SemEval2016TaskA,
    "cola": cola,
    "deisear": deisear,
    "sab": sab,
    "sb10k": sb10k,
    "subj": subj,
    "zsb": zsb,
}


@no_type_check
def load_dataset(
    name: str,
    config: Optional[str] = None,
) -> Mapping[str, datasets.Dataset]:
    if name == "yahoo":
        name = "zsb"
        config = "topic"
    if name == "unified":
        name = "zsb"
        config = "emotion"
    try:
        ds: Mapping[str, datasets.Dataset] = datasets.load_dataset(
            _modules[name].__file__,
            config,
            ignore_verifications=True,
        )
    except KeyError:
        ds: Mapping[str, datasets.Dataset] = datasets.load_dataset(
            name,
            config,
            ignore_verifications=True,
        )

    for split_name in ds:

        data = ds[split_name]

        if name == "head_qa":
            data = data.remove_columns(
                ["name", "year", "qid", "ra", "image", "answers"]
            )
            data = data.rename_column("category", "label")
            data = data.rename_column("qtext", "text")
            new_features = data.features.copy()
            new_features["label"] = datasets.ClassLabel(
                names=[
                    "medicine",
                    "nursery",
                    "chemistry",
                    "biology",
                    "psychology",
                    "pharmacology",
                ]
            )

            def to_label(d):
                d["label"] = new_features["label"].str2int(d["label"])
                return d

            data = data.map(
                to_label,
            )
            data = data.cast(new_features)
        elif name == "trec":
            data = data.remove_columns(["label-fine"])
            data = data.rename_column("label-coarse", "label")
        elif name == "amazon_reviews_multi":
            data = data.remove_columns(
                [
                    "review_id",
                    "product_id",
                    "reviewer_id",
                    "review_title",
                    "language",
                    "product_category",
                ]
            )
            data = data.rename_column("review_body", "text")
            data = data.rename_column("stars", "label")
            new_features = data.features.copy()
            new_features["label"] = datasets.ClassLabel(
                names=[
                    "1",
                    "2",
                    "3",
                    "4",
                    "5",
                ]
            )

            def to_label(d):
                d["label"] = d["label"] - 1
                return d

            data = data.map(
                to_label,
            )
            data = data.cast(new_features)

        if "label" not in data.features:
            raise ValueError(name, data.features.keys())

        if not isinstance(data.features["label"], datasets.ClassLabel):
            raise ValueError(name, data.features.keys())

        if "text" not in data.features:
            raise ValueError(name, data.features.keys())

        if (
            not isinstance(data.features["text"], datasets.Value)
            or data.features["text"].dtype != "string"  # noqa: W503
        ):
            raise ValueError(name, data.features.keys())

        ds[split_name] = data

    return ds


def get_english_datasets() -> List[Tuple[str, Optional[str]]]:
    return [  # type: ignore
        (name, None)
        for name in [
            "ag_news",
            "yahoo",
            "imdb",
            "yelp_review_full",
            "yelp_polarity",
            "SemEval2016TaskA",
            "unified",
            "cola",
            "subj",
            "trec",
        ]
    ] + [
        ("amazon_reviews_multi", "en")  # type: ignore
    ]


def get_german_datasets() -> List[Tuple[str, Optional[str]]]:
    return [  # type: ignore
        (name, None) for name in ["gnad10", "sb10k", "deisear"]
    ] + [
        ("amazon_reviews_multi", "de")  # type: ignore
    ]


def get_spanish_datasets() -> List[Tuple[str, Optional[str]]]:
    return [(name, None) for name in ["head_qa", "sab"]] + [  # type: ignore
        ("amazon_reviews_multi", "es")  # type: ignore
    ]
