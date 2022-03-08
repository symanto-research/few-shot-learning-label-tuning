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

import os
import tempfile
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import unittest

from symanto_fsb.experiments import get_results, run_experiment
from symanto_fsb.models.predictors.dual_encoder import DualEncoderPredictor
from symanto_fsb.models.siamese_models import label_tuning


def get_encoder() -> DualEncoderPredictor:
    return DualEncoderPredictor(
        {
            "model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            "module": "sentence_transformer",
        }
    )


class TestSentenceTransformer(unittest.TestCase):
    def test_dual_encoder_few_shot(self):
        get_config_backup = label_tuning._get_configs
        configs = label_tuning._get_configs()
        label_tuning._get_configs = lambda: configs[:1]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            run_experiment(
                predictor_fn=get_encoder,
                output_dir=temp_path,
                n_examples=4,
                n_trials=1,
                dataset="gnad10",
                config=None,
                n_test_examples=10,
            )

            results = get_results(temp_path)
            self.assertEqual(1, len(results))
            result = results[0]
            self.assertGreater(result["acc"], 59)
            self.assertGreater(result["mf1"], 59)

        label_tuning._get_configs = get_config_backup

    def test_dual_encoder_zero_shot(self):
        get_config_backup = label_tuning._get_configs
        configs = label_tuning._get_configs()
        label_tuning._get_configs = lambda: configs[:1]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            run_experiment(
                predictor_fn=get_encoder,
                output_dir=temp_path,
                n_examples=0,
                n_trials=1,
                dataset="gnad10",
                config=None,
                n_test_examples=10,
            )

            results = get_results(temp_path)
            self.assertEqual(1, len(results))
            result = results[0]
            self.assertGreater(result["acc"], 35)
            self.assertGreater(result["mf1"], 31)

        label_tuning._get_configs = get_config_backup


if __name__ == "__main__":
    unittest.main()
