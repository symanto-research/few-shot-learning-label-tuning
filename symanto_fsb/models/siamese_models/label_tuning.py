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

import collections
from typing import List, Optional

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score


_NEG_INFINITY = tf.float32.min


def _mf1(x, y):
    return f1_score(x, y, average="macro") * 100.0


def _get_loss(
    text_embeddings,
    text_labels,
    label_embeddings,
    dropout: float,
    loss_type: str = "softmax",
    epsilon: float = 1.0,
):
    """
    Args:
        text_embeddings: float[N, k]
        text_labels: float[N,K]
        label_embeddings: float[K, k]
        dropout: Dropout rate for class embeddings. 0.0 for none.
        loss_type: When BatchSoftmax, use sum_loss instead of margin loss.
        epsilon: Used for margin loss
    """
    tf.debugging.assert_all_finite(text_embeddings, "Text embeddings is NaN")
    tf.debugging.assert_all_finite(label_embeddings, "Label embeddings is NaN")
    # 1. Compute dot product between every text and every class.
    embedding_dim = label_embeddings.shape[1]
    if dropout > 0.0:
        r = tf.random.uniform(shape=(embedding_dim,))
        dp_mask = tf.where(r > dropout, tf.ones_like(r), tf.zeros_like(r))
        label_embeddings = label_embeddings * tf.expand_dims(dp_mask, axis=0)
    tf.debugging.assert_all_finite(label_embeddings, "Label embeddings is NaN")
    # float[N,K]
    dot_products = tf.matmul(text_embeddings, tf.transpose(label_embeddings))
    tf.debugging.assert_all_finite(dot_products, "Dot product is NaN")

    if loss_type == "softmax":
        if len(text_labels.shape) == 1:
            loss_per_example = tf.nn.sparse_softmax_cross_entropy_with_logits(
                text_labels, dot_products
            )
        else:
            loss_per_example = tf.nn.softmax_cross_entropy_with_logits(
                text_labels, dot_products
            )
    else:
        # 2. For every example, get the dot product of the correct class.
        n_classes = label_embeddings.shape[0]
        # float[N,K]
        # (1 if the class is correct, 0 otherwise)
        correct_class = np.argmax(text_labels, axis=-1)
        class_mask = tf.one_hot(
            correct_class, n_classes
        )  # pylint: disable=no-value-for-parameter
        # float[N]
        correct_dots = tf.reduce_sum(dot_products * class_mask, axis=-1)

        # 3. For every example, get the highest dot product of an incorrect class.
        if loss_type == "margin":
            # float[N,K]
            mask_dot_products = dot_products + (class_mask * _NEG_INFINITY)
            # float[N]
            wrong_dots = tf.reduce_max(mask_dot_products, axis=-1)
        elif loss_type == "sum":
            num_classes = label_embeddings.shape[0]
            wrong_dots = (
                tf.reduce_sum(dot_products, axis=-1) - correct_dots
            ) / (num_classes - 1)
        else:
            raise NotImplementedError(loss_type)

        # 4. Compute the difference per example.
        loss_per_example = correct_dots - wrong_dots
        # float[N]
        if loss_type == "margin":
            loss_per_example = tf.minimum(loss_per_example, epsilon)
        loss_per_example = -loss_per_example

    tf.debugging.assert_all_finite(loss_per_example, "Loss per example is NaN")
    return tf.reduce_mean(loss_per_example)


def label_tuning(
    text_embeddings,
    text_labels,
    label_embeddings,
    n_steps: int,
    reg_coefficient: float,
    learning_rate: float,
    dropout: float,
    **kwargs,
) -> np.ndarray:
    """
    With N as number of examples, K as number of classes, k as embedding dimension.

    Args:
        'text_embeddings': float[N,k] of embedded texts
        'text_labels': float[N,K] class score for each example.
        'label_embeddings': float[K,k] class embeddings
    Returns:
        float[K,k] updated class embeddings
    """
    if text_embeddings.shape[0] == 0:
        raise ValueError(text_embeddings.shape)
    if label_embeddings.shape[0] == 0:
        raise ValueError(label_embeddings.shape)

    text_embeddings = tf.constant(text_embeddings)
    text_labels = tf.constant(text_labels)
    label_embeddings = tf.constant(label_embeddings)

    init_label_embeddings = label_embeddings

    for i in range(n_steps):
        with tf.GradientTape() as tape:
            tape.watch(label_embeddings)
            dot_loss = _get_loss(
                text_embeddings,
                text_labels,
                label_embeddings,
                dropout=dropout,
                **kwargs,
            )
            drift_loss = tf.reduce_mean(
                (label_embeddings - init_label_embeddings) ** 2
            )
            total_loss = dot_loss + reg_coefficient * drift_loss
            gradient = tape.gradient(total_loss + drift_loss, label_embeddings)
            label_embeddings = label_embeddings - (learning_rate * gradient)

    label_embeddings = label_embeddings.numpy()
    return label_embeddings


def _get_configs():
    configs = [{}]
    param_spaces = {
        "dropout": [0.01, 0.1],
        "learning_rate": [0.5, 0.05],
        "reg_coefficient": [0.1, 0.01],
        "n_steps": [1000, 2000],
    }
    for param, space in param_spaces.items():
        new_configs = []
        for config in configs:
            for value in space:
                config = config.copy()
                config[param] = value
                new_configs.append(config)
        configs = new_configs
    return configs


def macro_f1_score(text_embeddings, text_labels, label_embeddings):
    scores = np.inner(text_embeddings, label_embeddings)
    predictions = scores.argmax(axis=-1)
    references = text_labels.argmax(axis=-1)
    return _mf1(references, predictions)


def _get_complement(folds: List[List[int]], fold: List[int]) -> List[int]:
    complement = set()
    for other_fold in folds:
        complement.update(other_fold)
    complement.difference_update(fold)
    return sorted(complement)


def _evaluate_hparams(
    text_embeddings, text_labels, label_embeddings, folds, loss_type, hparams
):
    predictions = []
    references = []
    for fold_index, test_indexes in enumerate(folds):
        train_indexes = _get_complement(folds, test_indexes)
        assert len(train_indexes) + len(test_indexes) == len(text_labels)
        tr_text_embeddings = text_embeddings[train_indexes]
        tr_text_labels = text_labels[train_indexes]
        new_label_embeddings = label_tuning(
            tr_text_embeddings,
            tr_text_labels,
            label_embeddings,
            loss_type=loss_type,
            **hparams,
        )
        te_text_embeddings = text_embeddings[test_indexes]
        scores = np.inner(te_text_embeddings, new_label_embeddings)
        predictions.append(scores.argmax(-1))
        references.append(text_labels[test_indexes].argmax(-1))
    references = np.concatenate(references, axis=0)
    predictions = np.concatenate(predictions, axis=0)
    mf1 = _mf1(references, predictions)
    return mf1


def find_hparams(
    text_embeddings,
    text_labels,
    label_embeddings,
    num_folds: int,
    loss_type: str = "softmax",
    configs: Optional[List[dict]] = None,
    n_workers: int = 1,
) -> dict:

    index_by_class = collections.defaultdict(list)
    for index, scores in enumerate(text_labels):
        class_index = np.argmax(scores)
        index_by_class[class_index].append(index)

    folds: List[List[int]] = [[] for _ in range(num_folds)]
    for indexes in index_by_class.values():
        for pos, index in enumerate(indexes):
            folds[pos % num_folds].append(index)

    initial_mf1 = macro_f1_score(text_embeddings, text_labels, label_embeddings)

    if configs is None:
        configs = _get_configs()

    if n_workers <= 1:
        mf1s = []
        for index, hparams in enumerate(configs):
            mf1 = _evaluate_hparams(
                text_embeddings,
                text_labels,
                label_embeddings,
                folds,
                loss_type,
                hparams,
            )
            mf1s.append(mf1)
    else:
        from concurrent.futures import ProcessPoolExecutor

        mf1s_futures = []
        exec = ProcessPoolExecutor(n_workers)
        for hparams in configs:
            mf1s_futures.append(
                exec.submit(
                    _evaluate_hparams,
                    text_embeddings,
                    text_labels,
                    label_embeddings,
                    folds,
                    loss_type,
                    hparams,
                )
            )

        mf1s = []
        for index, mf1 in enumerate(mf1s_futures):
            mf1 = mf1.result()
            mf1s.append(mf1)

    best_config_index = np.argmax(mf1s)
    best_config = configs[best_config_index]
    return {
        "hparams": best_config,
        "zero-shot macro f1": round(initial_mf1, 1),
        "few-shot macro f1": round(mf1s[best_config_index], 1),
    }
