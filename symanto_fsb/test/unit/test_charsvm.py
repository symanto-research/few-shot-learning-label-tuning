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

import tempfile
import unittest
from pathlib import Path

from symanto_fsb.experiments import get_results, run_experiment
from symanto_fsb.models.predictors.char_svm import CharSvmPredictor


class TestCharSvm(unittest.TestCase):
    def test_charsvm(self):
        with tempfile.TemporaryDirectory() as temp_dir:

            temp_path = Path(temp_dir)

            run_experiment(
                predictor_fn=lambda: CharSvmPredictor(),
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
            self.assertGreater(result["acc"], 35)
            self.assertGreater(result["mf1"], 36)


if __name__ == "__main__":
    unittest.main()
