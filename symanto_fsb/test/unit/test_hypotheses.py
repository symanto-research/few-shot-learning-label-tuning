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

from symanto_fsb.labels import to_hypothesis


class TestHypotheses(unittest.TestCase):
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
        texts = [to_hypothesis(label, "gnad10") for label in labels]
        self.assertEqual(
            texts,
            [
                "Das ist ein Artikel aus der Rubrik Web.",
                "Das ist ein Artikel aus der Rubrik Panorama.",
                "Das ist ein Artikel aus der Rubrik International.",
                "Das ist ein Artikel aus der Rubrik Wirtschaft.",
                "Das ist ein Artikel aus der Rubrik Sport.",
                "Das ist ein Artikel aus der Rubrik Inland.",
                "Das ist ein Artikel aus der Rubrik Etat.",
                "Das ist ein Artikel aus der Rubrik Wissenschaft.",
                "Das ist ein Artikel aus der Rubrik Kultur.",
            ],
        )

    def test_agnews(self):
        labels = ["World", "Sports", "Business", "Sci/Tech"]
        texts = [to_hypothesis(label, "ag_news") for label in labels]
        self.assertEqual(
            texts,
            [
                "It is world news.",
                "It is sports news.",
                "It is business news.",
                "It is science news.",
            ],
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
        texts = [to_hypothesis(label, "head_qa") for label in labels]
        self.assertEqual(
            texts,
            [
                "Está relacionado con la medicina.",
                "Está relacionado con la enfermería.",
                "Está relacionado con la química.",
                "Está relacionado con la biología.",
                "Está relacionado con la psicología.",
                "Está relacionado con la farmacología.",
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
        texts = [to_hypothesis(label, "yahoo") for label in labels]
        self.assertEqual(
            texts,
            ["It is related with " + label.lower() + "." for label in labels],
        )

    def test_amazon_reviews_multi(self):
        labels = ["1", "2", "3", "4", "5"]
        for lang in ["de", "es", "en"]:
            texts = [
                to_hypothesis(label, "amazon_reviews_multi", lang)
                for label in labels
            ]
            self.assertEqual(
                texts,
                {
                    "de": [
                        "Dieses Produkt ist furchtbar.",
                        "Dieses Produkt ist schlecht.",
                        "Dieses Produkt ist ok.",
                        "Dieses Produkt ist gut.",
                        "Dieses Produkt ist exzellent.",
                    ],
                    "es": [
                        "Este producto está terrible.",
                        "Este producto está mal.",
                        "Este producto está regular.",
                        "Este producto está bien.",
                        "Este producto está excelente.",
                    ],
                    "en": [
                        "This product is terrible.",
                        "This product is bad.",
                        "This product is okay.",
                        "This product is good.",
                        "This product is excellent.",
                    ],
                }[lang],
            )

    def test_imdb(self):
        labels = ["neg", "pos"]
        texts = [to_hypothesis(label, "imdb") for label in labels]
        self.assertEqual(texts, ["It was terrible.", "It was great."])

    def test_yelp_review_full(self):
        labels = ["1 star", "2 star", "3 stars", "4 stars", "5 stars"]
        texts = [to_hypothesis(label, "yelp_review_full") for label in labels]
        self.assertEqual(
            texts,
            [
                "It was terrible.",
                "It was bad.",
                "It was okay.",
                "It was good.",
                "It was great.",
            ],
        )

    def test_yelp_polarity(self):
        labels = ["1", "2"]
        texts = [to_hypothesis(label, "yelp_polarity") for label in labels]
        self.assertEqual(texts, ["It was terrible.", "It was great."])

    def test_sab(self):
        labels = [
            "positive",
            "neutral",
            "negative",
        ]
        texts = [to_hypothesis(label, "sab") for label in labels]
        self.assertEqual(
            texts,
            [
                "Esta persona expresa un sentimiento positivo.",
                "Esta persona expresa un sentimiento neutro.",
                "Esta persona expresa un sentimiento negativo.",
            ],
        )

    def test_SemEval2016TaskA(self):
        labels = [
            "positive",
            "negative",
            "neutral",
        ]
        texts = [to_hypothesis(label, "SemEval2016TaskA") for label in labels]
        self.assertEqual(
            texts,
            [
                "This person expresses a positive feeling.",
                "This person expresses a negative feeling.",
                "This person expresses a neutral feeling.",
            ],
        )

    def test_sb10k(self):
        labels = [
            "positive",
            "negative",
            "neutral",
        ]
        texts = [to_hypothesis(label, "sb10k") for label in labels]
        self.assertEqual(
            texts,
            [
                "Diese Person drückt ein positives Gefühl aus.",
                "Diese Person drückt ein negatives Gefühl aus.",
                "Diese Person drückt ein neutrales Gefühl aus.",
            ],
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
        texts = [to_hypothesis(label, "unified") for label in labels]
        self.assertEqual(
            texts,
            [
                "This person feels anger.",
                "This person feels disgust.",
                "This person feels fear.",
                "This person feels guilt.",
                "This person feels joy.",
                "This person feels love.",
                "This person doesn't feel any particular emotion.",
                "This person feels sadness.",
                "This person feels shame.",
                "This person feels surprise.",
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
        texts = [to_hypothesis(label, "deisear") for label in labels]
        self.assertEqual(
            texts,
            [
                "Diese Person empfindet Schuld.",
                "Diese Person empfindet Wut.",
                "Diese Person empfindet Ekel.",
                "Diese Person empfindet Angst.",
                "Diese Person empfindet Freude.",
                "Diese Person empfindet Scham.",
                "Diese Person empfindet Traurigkeit.",
            ],
        )

    def test_cola(self):
        labels = ["unacceptable", "acceptable"]
        texts = [to_hypothesis(label, "cola") for label in labels]
        self.assertEqual(texts, ["It is incorrect.", "It is correct."])

    def test_subj(self):
        labels = [
            "subjective",
            "objective",
        ]
        texts = [to_hypothesis(label, "subj") for label in labels]
        self.assertEqual(texts, ["It is subjective.", "It is objective."])

    def test_trec(self):
        labels = ["DESC", "ENTY", "ABBR", "HUM", "NUM", "LOC"]
        texts = [to_hypothesis(label, "trec") for label in labels]
        self.assertEqual(
            texts,
            [
                "It is description.",
                "It is entity.",
                "It is expression.",
                "It is human.",
                "It is number.",
                "It is location.",
            ],
        )


if __name__ == "__main__":
    unittest.main()
