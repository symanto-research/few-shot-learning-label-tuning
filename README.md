# Few-Shot Learning with Siamese Networks and Label Tuning
A few-shot learning method based on siamese networks.

Code & models for the [paper](https://openreview.net/forum?id=za_XIJLkkB8) to appear at [ACL 2022](https://www.2022.aclweb.org/).

# The Symanto Few-Shot Benchmark

`symanto-fsb` implements the benchmark discussed in ["Few-Shot Learning with Siamese Networks and Label Tuning"](https://openreview.net/forum?id=DU5CNf4sopk).

It can be easily extended to evaluate new models.
See the extension section below.

## Installation

### From source:

```console
pip install -e .
```

## CLI

### Char-SVM

```console
symanto-fsb evaluate-char-svm output/char_svm
```

### Sentence-Transformer

#### Zero-Shot

```console
symanto-fsb \
   evaluate-sentence-transformer \
    output/pml-mpnet \
    --gpu 0 \
    --n-examples=0
```

#### Few-Shot

```console
symanto-fsb \
   evaluate-sentence-transformer \
    output/pml-mpnet \
    --gpu 0 \
    --n-examples=8
```

## Extension

In general a new model is added by adding:

1. A new implementation of the [Predictor](symanto_fsb/models/predictors/__init__.py) interface
2. A new command to [cli.py](symanto_fsb/cli.py)


## Testing & Maintenance

```console
pip install -r dev-requirements.txt
dev-tools/format.sh
dev-tools/lint.sh
dev-tools/test.sh
```

## Models

For the sake of comparability we trained 4 models both English and multi-lingual (ML).


### Cross Attention

* [mpnet-base-snli-mnli](https://huggingface.co/symanto/mpnet-base-snli-mnli) (English)
* [xlm-roberta-base-snli-mnli-anli-xnli](https://huggingface.co/symanto/xlm-roberta-base-snli-mnli-anli-xnli) (ML)

### Siamese Networks

* [sn-mpnet-base-snli-mnli](https://huggingface.co/symanto/sn-mpnet-base-snli-mnli) (English)
* [sn-xlm-roberta-base-snli-mnli-anli-xnli](https://huggingface.co/symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli) (ML)

## Disclaimer

This is **not** an official Symanto product!

## How to Cite

```
@inproceedings{labeltuning2022,
	title        = {{Few-Shot} {Learning} with {Siamese} {Networks} and {Label} {Tuning}},
	author       = {M{\"u}ller, Thomas and Pérez-Torró, Guillermo and Franco-Salvador, Marc},
	year         = {2022},
	booktitle    = {ACL (to appear)},
	url          = {https://openreview.net/forum?id=za_XIJLkkB8},
}
```
