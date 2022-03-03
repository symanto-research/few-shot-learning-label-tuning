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

from symanto_fsb.labels import verbalize


class TestLabels(unittest.TestCase):
    def test_gnad(self):
        labels = [
            "Web",
            "Panorama",
            "International",
            "Wirtschaft",
            "Sport",
            "Inland",
            "Etat",
            "Wissenschaft",
            "Kultur",
        ]
        texts = [verbalize(label, "gnad10") for label in labels]
        self.assertEqual(
            texts,
            labels,
        )

    def test_agnews(self):
        labels = ["World", "Sports", "Business", "Sci/Tech"]
        texts = [verbalize(label, "ag_news") for label in labels]
        self.assertEqual(
            texts,
            ["world", "sports", "business", "science"],
        )

    def test_head_qa(self):
        labels = [
            "medicine",
            "nursery",
            "chemistry",
            "biology",
            "psychology",
            "pharmacology",
        ]
        texts = [verbalize(label, "head_qa") for label in labels]
        self.assertEqual(
            texts,
            [
                "medicina",
                "enfermería",
                "química",
                "biología",
                "psicología",
                "farmacología",
            ],
        )

    def test_yahoo(self):
        labels = [
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
        texts = [verbalize(label, "yahoo") for label in labels]
        self.assertEqual(texts, [label.lower() for label in labels])

    def test_amazon_reviews_multi(self):
        labels = ["1", "2", "3", "4", "5"]
        for lang in ["de", "es", "en"]:
            texts = [
                verbalize(label, "amazon_reviews_multi", lang)
                for label in labels
            ]
            self.assertEqual(
                texts,
                {
                    "de": ["furchtbar", "schlecht", "ok", "gut", "exzellent"],
                    "es": ["terrible", "mal", "regular", "bien", "excelente"],
                    "en": ["terrible", "bad", "okay", "good", "excellent"],
                }[lang],
            )

    def test_imdb(self):
        labels = ["neg", "pos"]
        texts = [verbalize(label, "imdb") for label in labels]
        self.assertEqual(
            texts,
            ["terrible", "great"],
        )

    def test_yelp_review_full(self):
        labels = ["1 star", "2 star", "3 stars", "4 stars", "5 stars"]
        texts = [verbalize(label, "yelp_review_full") for label in labels]
        self.assertEqual(texts, ["terrible", "bad", "okay", "good", "great"])

    def test_yelp_polarity(self):
        labels = ["1", "2"]
        texts = [verbalize(label, "yelp_polarity") for label in labels]
        self.assertEqual(
            texts,
            ["terrible", "great"],
        )

    def test_sab(self):
        labels = [
            "positive",
            "neutral",
            "negative",
        ]
        texts = [verbalize(label, "sab") for label in labels]
        self.assertEqual(
            texts,
            ["positivo", "neutro", "negativo"],
        )

    def test_SemEval2016TaskA(self):
        labels = [
            "positive",
            "negative",
            "neutral",
        ]
        texts = [verbalize(label, "SemEval2016TaskA") for label in labels]
        self.assertEqual(
            texts,
            ["positive", "negative", "neutral"],
        )

    def test_sb10k(self):
        labels = [
            "positive",
            "negative",
            "neutral",
        ]
        texts = [verbalize(label, "sb10k") for label in labels]
        self.assertEqual(
            texts,
            ["positiv", "negativ", "neutral"],
        )

    def test_unified(self):
        labels = [
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
        texts = [verbalize(label, "unified") for label in labels]
        self.assertEqual(
            texts,
            [
                "anger",
                "disgust",
                "fear",
                "guilt",
                "joy",
                "love",
                "no emotion",
                "sadness",
                "shame",
                "surprise",
            ],
        )

    def test_deisear(self):
        labels = [
            "guilt",
            "anger",
            "disgust",
            "fear",
            "joy",
            "shame",
            "sadness",
        ]
        texts = [verbalize(label, "deisear") for label in labels]
        self.assertEqual(
            texts,
            [
                "Schuld",
                "Wut",
                "Ekel",
                "Angst",
                "Freude",
                "Scham",
                "Traurigkeit",
            ],
        )

    def test_cola(self):
        labels = ["unacceptable", "acceptable"]
        texts = [verbalize(label, "cola") for label in labels]
        self.assertEqual(
            texts,
            ["incorrect", "correct"],
        )

    def test_subj(self):
        labels = [
            "subjective",
            "objective",
        ]
        texts = [verbalize(label, "subj") for label in labels]
        self.assertEqual(texts, ["subjective", "objective"])

    def test_trec(self):
        labels = ["DESC", "ENTY", "ABBR", "HUM", "NUM", "LOC"]
        texts = [verbalize(label, "trec") for label in labels]
        self.assertEqual(
            texts,
            [
                "description",
                "entity",
                "expression",
                "human",
                "number",
                "location",
            ],
        )


if __name__ == "__main__":
    unittest.main()
