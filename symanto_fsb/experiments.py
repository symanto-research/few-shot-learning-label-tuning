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

import json
from pathlib import Path
from typing import Callable, List, Mapping, Optional

import datasets
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from symanto_fsb.datasets import (
    get_english_datasets,
    get_german_datasets,
    get_spanish_datasets,
    load_dataset,
)
from symanto_fsb.labels import to_hypothesis
from symanto_fsb.models.predictors import Predictor
from symanto_fsb.sampling import sample


def macro_f1_score(x, y):
    return f1_score(x, y, average="macro")


def run_experiments(
    predictor_fn: Callable[[], Predictor],
    output_dir: Path,
    n_examples: int = 0,
    multilingual: bool = True,
    n_trials: int = 5,
    n_test_examples: Optional[int] = None,
):
    datasets = get_english_datasets()
    if multilingual:
        datasets.extend(get_german_datasets())
        datasets.extend(get_spanish_datasets())

    for dataset, config in datasets:
        run_experiment(
            predictor_fn,
            output_dir.joinpath(_get_dataset_dirname(dataset, config)),
            n_examples,
            n_trials,
            dataset,
            config,
            n_test_examples,
        )


def _get_dataset_dirname(dataset: str, config: Optional[str]) -> str:
    if config is None:
        return dataset
    return f"{dataset}_{config}"


def run_experiment(
    predictor_fn: Callable[[], Predictor],
    output_dir: Path,
    n_examples: int,
    n_trials: int,
    dataset: str,
    config: Optional[str],
    n_test_examples: Optional[int],
):

    data_dict = load_dataset(dataset, config)
    labels = data_dict["test"].features["label"]
    hypotheses = {
        label: to_hypothesis(label, dataset, config) for label in labels.names
    }

    if n_examples > 0:

        for trial in range(n_trials):
            result_file = output_dir.joinpath(
                f"results_{trial:02}"
            ).with_suffix(".json")
            if result_file.exists():
                continue

            predictor = predictor_fn()

            sample_data = sample(
                data_dict["train"], seed=trial, n_examples_per_label=n_examples
            )
            examples = [
                {
                    "text": text,
                    "label": labels.int2str(label),
                }
                for text, label in zip(
                    sample_data["text"], sample_data["label"]
                )
            ]
            predictor.few_shot(
                hypotheses=hypotheses,
                examples=examples,
            )

            test_model(
                predictor,
                hypotheses,
                data_dict["test"],
                result_file,
                n_test_examples,
            )

    else:

        result_file = output_dir.joinpath("results_00").with_suffix(".json")
        if result_file.exists():
            return
        predictor = predictor_fn()
        test_model(
            predictor,
            hypotheses,
            data_dict["test"],
            result_file,
            n_test_examples,
        )


def test_model(
    predictor: Predictor,
    hypotheses: Mapping[str, str],
    test_data: datasets.Dataset,
    result_file: Path,
    n_test_examples: Optional[int],
):
    labels = test_data.features["label"]

    if n_test_examples is not None:
        test_data = sample(  # type: ignore
            test_data, seed=42, n_examples_per_label=n_test_examples
        )

    examples = [
        {
            "text": text,
            "label": None,
        }
        for text in test_data["text"]
    ]

    predictions = predictor.predict(hypotheses, examples)

    result_file.parent.mkdir(parents=True, exist_ok=True)

    pred_file = result_file.with_name(
        result_file.stem + "_predictions"
    ).with_suffix(".jsonl")
    with pred_file.open("tw") as writer:
        for text, pred, ref in zip(
            test_data["text"],
            predictions,
            test_data["label"],
        ):
            pred_label = labels.int2str(pred)
            ref_label = labels.int2str(ref)
            writer.write(
                json.dumps(
                    {
                        "text": text,
                        "reference": ref_label,
                        "prediction": pred_label,
                    }
                )
            )
            writer.write("\n")

    acc = accuracy_score(test_data["label"], predictions)
    mf1 = macro_f1_score(test_data["label"], predictions)

    with result_file.open("tw") as writer:
        json.dump(
            {
                "acc": acc * 100.0,
                "mf1": mf1 * 100.0,
            },
            writer,
            indent="  ",
        )


def get_results(data_dir: Path) -> List[dict]:
    results = []
    for filepath in data_dir.glob("results_??.json"):
        with filepath.open("tr") as reader:
            result = json.load(reader)
            if not isinstance(result, dict):
                raise ValueError(result)
        results.append(result)
    return results


def report(data_dir: Path, output_file: Path) -> None:
    result_files = data_dir.glob("**/results_00.json")

    rows = []

    for result_file in result_files:
        result_dir = result_file.parent
        dataset = result_dir.name
        n_examples = result_dir.parent.name
        model = result_dir.parent.parent.name

        results = get_results(result_dir)

        frame = pd.DataFrame(results)

        mean_results = frame.mean()

        row = {
            "model": model,
            "n": n_examples,
            "dataset": dataset,
        }

        row.update(
            {
                metric: round(value, 1)
                for metric, value in dict(mean_results).items()
            }
        )

        rows.append(row)

    frame = pd.DataFrame(rows)
    frame.sort_values("dataset", inplace=True)

    rows = []
    for (model, n_examples), model_data in frame.groupby(["model", "n"]):

        row = {
            "model": model,
            "n": n_examples,
        }

        for _, model_row in model_data.iterrows():
            row[model_row["dataset"]] = model_row["mf1"]

        rows.append(row)

    frame = pd.DataFrame(rows)
    with output_file.open("tw") as writer:
        frame.to_csv(writer, index=False, sep="\t")
