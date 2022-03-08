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

from typing import Optional


def verbalize(label: str, dataset: str, config: Optional[str] = None) -> str:
    if dataset == "gnad10":
        return label
    if dataset == "ag_news":
        return {
            "world": "world",
            "sports": "sports",
            "business": "business",
            "sci/tech": "science",
        }[label.lower()]
    if dataset == "head_qa":
        return {
            "medicine": "medicina",
            "nursery": "enfermería",
            "chemistry": "química",
            "biology": "biología",
            "psychology": "psicología",
            "pharmacology": "farmacología",
        }[label]
    if dataset == "yahoo":
        return label.lower()
    if dataset == "amazon_reviews_multi":
        if config == "de":
            return {
                "1": "furchtbar",
                "2": "schlecht",
                "3": "ok",
                "4": "gut",
                "5": "exzellent",
            }[label]
        if config == "en":
            return {
                "1": "terrible",
                "2": "bad",
                "3": "okay",
                "4": "good",
                "5": "excellent",
            }[label]
        if config == "es":
            return {
                "1": "terrible",
                "2": "mal",
                "3": "regular",
                "4": "bien",
                "5": "excelente",
            }[label]
        else:
            raise NotImplementedError(dataset, config)
    if dataset == "imdb":
        return {
            "pos": "great",
            "neg": "terrible",
        }[label]
    if dataset == "yelp_review_full":
        return {
            "1 star": "terrible",
            "2 star": "bad",
            "3 stars": "okay",
            "4 stars": "good",
            "5 stars": "great",
        }[label]
    if dataset == "yelp_polarity":
        return {
            "2": "great",
            "1": "terrible",
        }[label]
    if dataset == "sab":
        return {
            "positive": "positivo",
            "neutral": "neutro",
            "negative": "negativo",
        }[label]
    if dataset == "SemEval2016TaskA":
        return {
            "positive": "positive",
            "neutral": "neutral",
            "negative": "negative",
        }[label]
    if dataset == "unified":
        if label == "noemo":
            return "no emotion"
        return label
    if dataset == "deisear":
        return {
            "guilt": "Schuld",
            "anger": "Wut",
            "disgust": "Ekel",
            "fear": "Angst",
            "joy": "Freude",
            "shame": "Scham",
            "sadness": "Traurigkeit",
        }[label]
    if dataset == "cola":
        return {
            "acceptable": "correct",
            "unacceptable": "incorrect",
        }[label]
    if dataset == "subj":
        return {
            "objective": "objective",
            "subjective": "subjective",
        }[label]
    if dataset == "trec":
        return {
            "ABBR": "expression",
            "DESC": "description",
            "ENTY": "entity",
            "HUM": "human",
            "LOC": "location",
            "NUM": "number",
        }[label]
    if dataset == "sitatuion":
        return {
            "food": "people there need food.",
            "infra": "people there need infrastructures.",
            "med": "people need medical assistance.",
            "search": "people there need search.",
            "shelter": "people there need shelter.",
            "utils": "people there need utilities.",
            "water": "people there need water.",
            "crimeviolence": "crime violence happened there.",
            "terrorism": "this text describes terrorist activity.",
            "evac": "This place is very dangerous and it is urgent to evacuate people to safety.",
            "regimechange": "Regime change happened in this country.",
            "out-of-domain": "This about something else.",
        }[label]
    if dataset == "sb10k":
        return {
            "positive": "positiv",
            "neutral": "neutral",
            "negative": "negativ",
        }[label]
    raise NotImplementedError(dataset, config)


def to_hypothesis(
    label: str, dataset: str, config: Optional[str] = None
) -> str:
    label = verbalize(label, dataset, config)
    if dataset == "gnad10":
        return f"Das ist ein Artikel aus der Rubrik {label}."
    if dataset == "ag_news":
        return f"It is {label} news."
    if dataset == "head_qa":
        return f"Está relacionado con la {label}."
    if dataset == "yahoo":
        return f"It is related with {label}."
    if dataset == "amazon_reviews_multi":
        if config == "de":
            return f"Dieses Produkt ist {label}."
        if config == "es":
            return f"Este producto está {label}."
        if config == "en":
            return f"This product is {label}."
        raise NotImplementedError(dataset, config)
    if dataset == "imdb":
        return f"It was {label}."
    if dataset == "yelp_review_full":
        return f"It was {label}."
    if dataset == "yelp_polarity":
        return f"It was {label}."
    if dataset == "sab":
        return f"Esta persona expresa un sentimiento {label}."
    if dataset == "sb10k":
        return f"Diese Person drückt ein {label}es Gefühl aus."
    if dataset == "SemEval2016TaskA":
        return f"This person expresses a {label} feeling."
    if dataset == "unified":
        if label == "no emotion":
            return "This person doesn't feel any particular emotion."
        return f"This person feels {label}."
    if dataset == "deisear":
        return f"Diese Person empfindet {label}."
    if dataset in ["trec", "subj", "cola"]:
        return f"It is {label}."
    if dataset == "situation":
        return label
    raise NotImplementedError(dataset, config)
