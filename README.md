# Few-Shot Learning with Siamese Networks and Label Tuning
A few-shot learning method based on siamese networks.

Code & models for the [paper](https://openreview.net/forum?id=za_XIJLkkB8) to appear at [ACL 2022](https://www.2022.aclweb.org/).

**Code coming soon ...**

## Models

For the sake of comparability we trained 4 models both English and multi-lingual (ML).


### Cross Attention

* [mpnet-base-snli-mnli](https://huggingface.co/symanto/mpnet-base-snli-mnli) (English)
* [xlm-roberta-base-snli-mnli-anli-xnli](https://huggingface.co/symanto/xlm-roberta-base-snli-mnli-anli-xnli) (ML)

### Siamese Networks

* [sn-mpnet-base-snli-mnli](https://huggingface.co/symanto/sn-mpnet-base-snli-mnli) (English)
* [sn-xlm-roberta-base-snli-mnli-anli-xnli](https://huggingface.co/symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli) (ML)

Note that we recommend to use these [Sentence Transformer](sbert.net) models in real world applications:

* [paraphrase-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-mpnet-base-v2) (English)
* [paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2) (ML)

## Disclaimer

This is **not** an official Symanto product!
