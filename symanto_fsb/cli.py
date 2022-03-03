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

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer()


@app.command()
def evaluate_char_svm(
    output_dir: Path,
    n_examples: int = 8,
    multilingual: bool = True,
    n_test_examples: Optional[int] = None,
):
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if n_examples <= 0:
        raise ValueError("SVM requires n_examples > 0!")

    from symanto_fsb.experiments import run_experiments
    from symanto_fsb.models.predictors.char_svm import CharSvmPredictor

    run_experiments(
        predictor_fn=lambda: CharSvmPredictor(),
        output_dir=output_dir.joinpath(f"{n_examples:03}"),
        n_examples=n_examples,
        multilingual=multilingual,
        n_test_examples=n_test_examples,
    )


@app.command()
def evaluate_sentence_transformer(
    output_dir: Path,
    model: str = "symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli",
    gpu: str = "",
    n_examples: int = 8,
    multilingual: bool = True,
    n_test_examples: Optional[int] = None,
):
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")

    from symanto_fsb.experiments import run_experiments
    from symanto_fsb.models.predictors.dual_encoder import DualEncoderPredictor

    run_experiments(
        predictor_fn=lambda: DualEncoderPredictor(
            {
                "module": "sentence_transformer",
                "model": model,
            }
        ),
        output_dir=output_dir.joinpath(f"{n_examples:03}"),
        n_examples=n_examples,
        multilingual=multilingual,
        n_test_examples=n_test_examples,
    )


@app.command()
def report(output_dir: Path, output_file: Path):
    from symanto_fsb.experiments import report as _report

    _report(output_dir, output_file)


if __name__ == "__main__":
    app()
