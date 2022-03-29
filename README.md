# Few-Shot Learning with Siamese Networks and Label Tuning
A few-shot learning method based on siamese networks.

Code & models for the [paper](https://arxiv.org/abs/2203.14655) to appear at [ACL 2022](https://www.2022.aclweb.org/).

# The Symanto Few-Shot Benchmark

`symanto-fsb` implements the benchmark discussed in the paper.

It can be easily extended to evaluate new models.
See the extension section below.

## Installation

```console
pip install -e .
```

## CLI

### Char-SVM

```console
symanto-fsb evaluate-char-svm output/char_svm --n-trials=1
```

This will run the specified number of trials on each dataset and write results to the output directory. Afterwards you can create a result table:

```console
symanto-fsb report output /tmp/report.tsv
```

### Sentence-Transformer

#### Zero-Shot

```console
symanto-fsb \
   evaluate-sentence-transformer \
    output/pml-mpnet \
    --gpu 0 \
    --n-examples=0 \
    --n-trials=1 
```

#### Few-Shot

```console
symanto-fsb \
   evaluate-sentence-transformer \
    output/pml-mpnet \
    --gpu 0 \
    --n-examples=8 \
    --n-trials=1
```

## Extension

In general a new model is added by adding:

1. A new implementation of the [Predictor](symanto_fsb/models/predictors/__init__.py) interface
2. A new command to [cli.py](symanto_fsb/cli.py)

## Known Issues

Datasets hosted on Google Drive do not work right now: [datasets issue/3809](https://github.com/huggingface/datasets/issues/3809)

## Testing & Maintenance

```console
pip install -r dev-requirements.txt
dev-tools/format.sh
dev-tools/lint.sh
dev-tools/test.sh
```

## Models

For the sake of comparability we trained 4.
The code above can be used with the Siamese network models.

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
	url          = {https://arxiv.org/abs/2203.14655},
}
```
