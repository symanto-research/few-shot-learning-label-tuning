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

import unittest

from symanto_fsb.datasets import load_dataset


class TestDatasets(unittest.TestCase):
    def test_gnad(self):
        dataset = load_dataset("gnad10")
        self.assertEqual(len(dataset["test"]), 1028)
        self.assertEqual(len(dataset["train"]), 9245)
        self.assertEqual(
            dataset["train"].features["label"].names,
            [
                "Web",
                "Panorama",
                "International",
                "Wirtschaft",
                "Sport",
                "Inland",
                "Etat",
                "Wissenschaft",
                "Kultur",
            ],
        )

    def test_agnews(self):
        dataset = load_dataset("ag_news")
        self.assertEqual(len(dataset["test"]), 7600)
        self.assertEqual(len(dataset["train"]), 120_000)
        self.assertEqual(
            dataset["train"].features["label"].names,
            ["World", "Sports", "Business", "Sci/Tech"],
        )

    # This sometimes fails with a checksum error.
    # Maybe a Google drive issue?
    # def test_head_qa(self):
    #     dataset = load_dataset("head_qa")
    #     self.assertEqual(len(dataset["test"]), 2742)
    #     self.assertEqual(
    #         len(dataset["train"]) + len(dataset["validation"]), 4023
    #     )
    #     self.assertEqual(
    #         dataset["train"].features["label"].names,
    #         [
    #             "medicine",
    #             "nursery",
    #             "chemistry",
    #             "biology",
    #             "psychology",
    #             "pharmacology",
    #         ],
    #     )

    # This sometimes fails with a checksum error.
    # Maybe a Google drive issue?
    # def test_yahoo(self):
    #     dataset = load_dataset("yahoo")
    #     self.assertEqual(len(dataset["test"]), 100_000)
    #     self.assertEqual(
    #         len(dataset["train"]) + len(dataset["validation"]), 1_360_000
    #     )
    #     self.assertEqual(
    #         dataset["train"].features["label"].names,
    #         [
    #             "Society & Culture",
    #             "Science & Mathematics",
    #             "Health",
    #             "Education & Reference",
    #             "Computers & Internet",
    #             "Sports",
    #             "Business & Finance",
    #             "Entertainment & Music",
    #             "Family & Relationships",
    #             "Politics & Government",
    #         ],
    #     )

    def test_amazon_reviews_multi(self):
        for lang in ["de", "es", "en"]:
            dataset = load_dataset("amazon_reviews_multi", config=lang)
            self.assertEqual(len(dataset["test"]), 5000)
            self.assertEqual(
                len(dataset["train"]) + len(dataset["validation"]), 205_000
            )
            self.assertEqual(
                dataset["train"].features["label"].names,
                ["1", "2", "3", "4", "5"],
            )

    def test_imdb(self):
        dataset = load_dataset("imdb")
        self.assertEqual(len(dataset["test"]), 25_000)
        self.assertEqual(len(dataset["train"]), 25_000)
        self.assertEqual(
            dataset["train"].features["label"].names, ["neg", "pos"]
        )

    # This sometimes fails with a checksum error.
    # Maybe a Google drive issue?
    # def test_yelp_review_full(self):
    #     dataset = load_dataset("yelp_review_full")
    #     self.assertEqual(len(dataset["test"]), 50_000)
    #     self.assertEqual(len(dataset["train"]), 650_000)
    #     self.assertEqual(
    #         dataset["train"].features["label"].names,
    #         ["1 star", "2 star", "3 stars", "4 stars", "5 stars"],
    #     )

    def test_yelp_polarity(self):
        dataset = load_dataset("yelp_polarity")
        self.assertEqual(len(dataset["test"]), 38_000)
        self.assertEqual(len(dataset["train"]), 560_000)
        self.assertEqual(dataset["train"].features["label"].names, ["1", "2"])

    def test_sab(self):
        dataset = load_dataset("sab")
        self.assertEqual(len(dataset["test"]), 459)
        self.assertEqual(
            len(dataset["train"]) + len(dataset["validation"]), 3979
        )
        self.assertEqual(
            dataset["train"].features["label"].names,
            [
                "positive",
                "neutral",
                "negative",
            ],
        )

    def test_SemEval2016TaskA(self):
        dataset = load_dataset("SemEval2016TaskA")
        self.assertEqual(len(dataset["test"]), 20_632)
        self.assertEqual(
            len(dataset["train"]) + len(dataset["validation"]), 9_834
        )
        self.assertEqual(
            dataset["train"].features["label"].names,
            [
                "positive",
                "negative",
                "neutral",
            ],
        )

    def test_sb10k(self):
        dataset = load_dataset("sb10k")
        self.assertEqual(len(dataset["test"]), 994)
        self.assertEqual(len(dataset["train"]), 8955)
        self.assertEqual(
            dataset["train"].features["label"].names,
            [
                "positive",
                "negative",
                "neutral",
            ],
        )

    # def test_unified(self):
    #     dataset = load_dataset("unified")
    #     self.assertEqual(len(dataset["test"]), 15689)
    #     self.assertEqual(
    #         len(dataset["train"]) + len(dataset["validation"]), 42145
    #     )
    #     self.assertEqual(
    #         dataset["train"].features["label"].names,
    #         [
    #             "anger",
    #             "disgust",
    #             "fear",
    #             "guilt",
    #             "joy",
    #             "love",
    #             "noemo",
    #             "sadness",
    #             "shame",
    #             "surprise",
    #         ],
    #     )

    def test_deisear(self):
        dataset = load_dataset("deisear")
        self.assertEqual(len(dataset["test"]), 340)
        self.assertEqual(
            len(dataset["train"]) + len(dataset["validation"]), 643
        )
        self.assertEqual(
            dataset["train"].features["label"].names,
            [
                "guilt",
                "anger",
                "disgust",
                "fear",
                "joy",
                "shame",
                "sadness",
            ],
        )

    def test_cola(self):
        dataset = load_dataset("cola")
        self.assertEqual(len(dataset["test"]), 1043)
        self.assertEqual(len(dataset["train"]), 8551)
        self.assertEqual(
            dataset["train"].features["label"].names,
            ["unacceptable", "acceptable"],
        )
        self.assertEqual(
            dataset["test"].features["label"].names,
            ["unacceptable", "acceptable"],
        )
        for label in dataset["test"]["label"]:
            self.assertIn(
                dataset["test"].features["label"].int2str(label),
                ["unacceptable", "acceptable"],
            )

    def test_subj(self):
        dataset = load_dataset("subj")
        self.assertEqual(len(dataset["test"]), 1981)
        self.assertEqual(
            len(dataset["train"]) + len(dataset["validation"]), 8019
        )
        self.assertEqual(
            dataset["train"].features["label"].names,
            [
                "subjective",
                "objective",
            ],
        )

    def test_trec(self):
        dataset = load_dataset("trec")
        self.assertEqual(len(dataset["test"]), 500)
        self.assertEqual(len(dataset["train"]), 5452)
        self.assertEqual(
            dataset["train"].features["label"].names,
            ["DESC", "ENTY", "ABBR", "HUM", "NUM", "LOC"],
        )


if __name__ == "__main__":
    unittest.main()
