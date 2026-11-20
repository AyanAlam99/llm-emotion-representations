"""Analysis of emotion vectors: held-out classification, cosine structure, PCA.

Every function here works on CPU tensors only. Nothing loads the model.

Three analyses, in order of evidential weight:

1. `held_out_classification` - vectors are built on 80% of the `pair_id`s and
   tested on the other 20%. This is the only result here that is not fitted on
   the data it is scored on.
2. `cosine_matrix` / `cosine_summary` - how distinct the 10 emotion directions
   are from each other. Jeong's metric is `mean_pairwise_cosine`.
3. `pca` - the low-dimensional structure of the 10 directions.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

import torch

from .config import TARGET_EMOTIONS
from .vectors import build_emotion_vectors

NEGATIVE_EMOTIONS = (
    "anger",
    "annoyance",
    "fear",
    "nervousness",
    "sadness",
    "disappointment",
)

POSITIVE_EMOTIONS = (
    "joy",
    "excitement",
    "admiration",
    "gratitude",
)

assert set(NEGATIVE_EMOTIONS) | set(POSITIVE_EMOTIONS) == set(TARGET_EMOTIONS)
assert not set(NEGATIVE_EMOTIONS) & set(POSITIVE_EMOTIONS)


def _finite(value: float) -> float | None:
    """Return a JSON-safe float, or None for nan/inf."""
    return float(value) if math.isfinite(value) else None


# ---------------------------------------------------------------- shared math


def stack_layer(
    vectors: dict[str, torch.Tensor],
    layer: int,
    emotions: Sequence[str],
) -> torch.Tensor:
    """Stack one layer of per-emotion vectors into a [len(emotions), hidden] matrix."""
    missing = [emotion for emotion in emotions if emotion not in vectors]
    if missing:
        raise KeyError(f"missing vectors for emotions: {missing}")
    return torch.stack([vectors[emotion][layer].float() for emotion in emotions])


def unit_rows(matrix: torch.Tensor) -> torch.Tensor:
    return matrix / matrix.norm(dim=-1, keepdim=True).clamp_min(1e-8)


# ----------------------------------------------------------- held-out testing


def split_pair_ids(
    records: Sequence[dict],
    *,
    seed: int,
    train_fraction: float = 0.8,
) -> tuple[list[str], list[str]]:
    """Split `pair_id`s into train and test groups.

    Splitting happens at the `pair_id` level, never at the row level: all 11
    stories of one group (1 neutral + 10 emotional) land on the same side, so a
    story can never be scored against a vector that saw its own matched neutral.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1), received {train_fraction}")

    pair_ids = sorted({record["pair_id"] for record in records})
    shuffled = list(pair_ids)
    random.Random(seed).shuffle(shuffled)
    cut = round(len(shuffled) * train_fraction)
    if cut == 0 or cut == len(shuffled):
        raise ValueError(f"split leaves one side empty: {len(shuffled)} pair_ids")
    return sorted(shuffled[:cut]), sorted(shuffled[cut:])


def _test_items(
    activations: dict[str, torch.Tensor],
    records: Sequence[dict],
    test_pair_ids: Sequence[str],
    emotions: Sequence[str],
    *,
    reference: str,
    train_mean: torch.Tensor | None,
) -> tuple[torch.Tensor, list[int]]:
    """Build the held-out test matrix.

    Returns `(differences, labels)` where `differences` is
    `[test_stories, layers, hidden]` and `labels` holds the index into
    `emotions` of the true label of each row.

    `reference="paired"` subtracts the matched neutral story of the same
    `pair_id`. `reference="train_mean"` subtracts the mean activation over all
    training stories, which is the variant that does not need the held-out
    neutral story.
    """
    wanted = set(test_pair_ids)
    index = {emotion: position for position, emotion in enumerate(emotions)}

    neutrals: dict[str, torch.Tensor] = {}
    targets: list[tuple[str, str, str]] = []
    for record in records:
        if record["pair_id"] not in wanted:
            continue
        if record["condition"] == "neutral":
            neutrals[record["pair_id"]] = activations[record["sample_id"]]
        elif record["emotion"] in index:
            targets.append(
                (record["pair_id"], record["emotion"], record["sample_id"])
            )

    rows: list[torch.Tensor] = []
    labels: list[int] = []
    for pair_id, emotion, sample_id in targets:
        activation = activations[sample_id].float()
        if reference == "paired":
            neutral = neutrals.get(pair_id)
            if neutral is None:
                continue
            rows.append(activation - neutral.float())
        elif reference == "train_mean":
            if train_mean is None:
                raise ValueError("train_mean reference requires a train_mean tensor")
            rows.append(activation - train_mean)
        else:
            raise ValueError(f"unknown reference: {reference!r}")
        labels.append(index[emotion])

    if not rows:
        raise RuntimeError("no held-out test stories were built")
    return torch.stack(rows), labels


def _accuracy_per_layer(
    differences: torch.Tensor,
    labels: Sequence[int],
    directions: torch.Tensor,
) -> tuple[list[float], torch.Tensor]:
    """Argmax-projection accuracy at every layer.

    `differences` is [stories, layers, hidden], `directions` is
    [layers, emotions, hidden] of unit rows. Returns per-layer accuracy and the
    [layers, stories] predicted label indices.
    """
    truth = torch.tensor(labels)
    layer_count = differences.shape[1]
    accuracies: list[float] = []
    predictions = torch.empty(layer_count, differences.shape[0], dtype=torch.long)
    for layer in range(layer_count):
        # [stories, hidden] @ [hidden, emotions] -> [stories, emotions]
        scores = differences[:, layer, :] @ directions[layer].T
        predicted = scores.argmax(dim=-1)
        predictions[layer] = predicted
        accuracies.append(float((predicted == truth).float().mean()))
    return accuracies, predictions


def _confusion(
    predicted: torch.Tensor,
    labels: Sequence[int],
    emotion_count: int,
) -> list[list[int]]:
    """Confusion counts, rows = true emotion, columns = predicted emotion."""
    matrix = torch.zeros(emotion_count, emotion_count, dtype=torch.long)
    for true_label, prediction in zip(labels, predicted.tolist(), strict=True):
        matrix[true_label, prediction] += 1
    return matrix.tolist()


def held_out_classification(
    activations: dict[str, torch.Tensor],
    records: Sequence[dict],
    emotions: Sequence[str],
    *,
    seed: int,
    train_fraction: float = 0.8,
    reference: str = "paired",
    random_repeats: int = 5,
) -> dict:
    """Build vectors on 80% of the groups, classify the held-out 20%.

    For every held-out emotional story the activation is turned into a
    difference vector, projected onto all 10 unit emotion directions, and the
    largest projection is the prediction. Chance is 1/len(emotions).
    """
    train_pair_ids, test_pair_ids = split_pair_ids(
        records, seed=seed, train_fraction=train_fraction
    )
    train_set = set(train_pair_ids)
    train_records = [
        record for record in records if record["pair_id"] in train_set
    ]

    _raw, unit, pair_ids = build_emotion_vectors(activations, train_records)
    missing = [emotion for emotion in emotions if emotion not in unit]
    if missing:
        raise RuntimeError(f"training split produced no vector for: {missing}")

    train_mean = None
    if reference == "train_mean":
        train_mean = torch.stack(
            [activations[record["sample_id"]].float() for record in train_records]
        ).mean(dim=0)

    differences, labels = _test_items(
        activations,
        records,
        test_pair_ids,
        emotions,
        reference=reference,
        train_mean=train_mean,
    )

    # [layers, emotions, hidden], unit rows so the argmax is a cosine argmax.
    layer_count, hidden_size = next(iter(unit.values())).shape
    directions = torch.stack(
        [unit_rows(stack_layer(unit, layer, emotions)) for layer in range(layer_count)]
    )

    accuracies, predictions = _accuracy_per_layer(differences, labels, directions)
    best_index = int(torch.tensor(accuracies).argmax())

    # Random-direction control: same pipeline, meaningless directions.
    generator = torch.Generator().manual_seed(seed)
    random_accuracies = []
    for _ in range(random_repeats):
        noise = unit_rows(torch.randn(len(emotions), hidden_size, generator=generator))
        control = noise.unsqueeze(0).expand(layer_count, -1, -1)
        control_accuracy, _ = _accuracy_per_layer(differences, labels, control)
        random_accuracies.append(control_accuracy)
    random_mean = [
        sum(values) / len(values) for values in zip(*random_accuracies, strict=True)
    ]

    per_layer = [
        {
            "layer": layer,
            "accuracy": _finite(accuracy),
            "random_direction_accuracy": _finite(random_mean[layer]),
        }
        for layer, accuracy in enumerate(accuracies)
    ]

    return {
        "reference": reference,
        "reference_definition": (
            "paired = held-out emotional story minus its matched neutral story; "
            "train_mean = held-out emotional story minus the mean training activation"
        ),
        "seed": seed,
        "train_fraction": train_fraction,
        "train_pair_count": len(train_pair_ids),
        "test_pair_count": len(test_pair_ids),
        "test_story_count": len(labels),
        "train_pair_counts_per_emotion": {
            emotion: len(ids) for emotion, ids in pair_ids.items()
        },
        "chance_accuracy": 1.0 / len(emotions),
        "best_layer": best_index,
        "best_layer_accuracy": _finite(accuracies[best_index]),
        "random_direction_repeats": random_repeats,
        "per_layer": per_layer,
        "accuracy_by_layer": [_finite(value) for value in accuracies],
        "_predictions": predictions,
        "_labels": labels,
    }


def per_emotion_accuracy(
    predicted: torch.Tensor,
    labels: Sequence[int],
    emotions: Sequence[str],
) -> dict[str, dict]:
    """Recall per true emotion at one layer."""
    totals = {emotion: 0 for emotion in emotions}
    correct = {emotion: 0 for emotion in emotions}
    for true_label, prediction in zip(labels, predicted.tolist(), strict=True):
        emotion = emotions[true_label]
        totals[emotion] += 1
        if prediction == true_label:
            correct[emotion] += 1
    return {
        emotion: {
            "correct": correct[emotion],
            "total": totals[emotion],
            "accuracy": _finite(correct[emotion] / totals[emotion])
            if totals[emotion]
            else None,
        }
        for emotion in emotions
    }


def confusion_matrix(
    predicted: torch.Tensor,
    labels: Sequence[int],
    emotions: Sequence[str],
) -> list[list[int]]:
    return _confusion(predicted, labels, len(emotions))


# --------------------------------------------------------- cosine structure


def cosine_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """Row-wise cosine similarity matrix. Row scaling does not change the result."""
    rows = unit_rows(matrix.float())
    return rows @ rows.T


def upper_triangle_values(matrix: torch.Tensor) -> torch.Tensor:
    """All off-diagonal values of a symmetric matrix, each pair counted once."""
    size = matrix.shape[0]
    indices = torch.triu_indices(size, size, offset=1)
    return matrix[indices[0], indices[1]]


def mean_pairwise_cosine(matrix: torch.Tensor) -> float:
    """Mean cosine over all distinct row pairs. Jeong's separation metric."""
    if matrix.shape[0] < 2:
        return float("nan")
    return float(upper_triangle_values(cosine_matrix(matrix)).mean())


def pair_values(
    cosine: torch.Tensor,
    emotions: Sequence[str],
    group_a: Sequence[str],
    group_b: Sequence[str] | None = None,
) -> torch.Tensor:
    """Cosines inside one group (group_b is None) or across two disjoint groups."""
    index = {emotion: position for position, emotion in enumerate(emotions)}
    rows = [index[emotion] for emotion in group_a if emotion in index]
    if group_b is None:
        values = [
            cosine[i, j]
            for position, i in enumerate(rows)
            for j in rows[position + 1 :]
        ]
    else:
        columns = [index[emotion] for emotion in group_b if emotion in index]
        values = [cosine[i, j] for i in rows for j in columns]
    if not values:
        return torch.zeros(0)
    return torch.stack(values)


def _group_mean(
    cosine: torch.Tensor,
    emotions: Sequence[str],
    group_a: Sequence[str],
    group_b: Sequence[str] | None = None,
) -> float:
    values = pair_values(cosine, emotions, group_a, group_b)
    return float(values.mean()) if values.numel() else float("nan")


def cosine_summary(cosine: torch.Tensor, emotions: Sequence[str]) -> dict:
    """Scalar summaries of one 10x10 cosine matrix."""
    pairs = upper_triangle_values(cosine)
    size = cosine.shape[0]
    indices = torch.triu_indices(size, size, offset=1)
    most_similar = int(pairs.argmax())
    least_similar = int(pairs.argmin())
    return {
        "mean_pairwise_cosine": _finite(float(pairs.mean())),
        "min_pairwise_cosine": _finite(float(pairs.min())),
        "max_pairwise_cosine": _finite(float(pairs.max())),
        "negative_cosine_pair_count": int((pairs < 0).sum()),
        "pair_count": int(pairs.numel()),
        "most_similar_pair": [
            emotions[int(indices[0][most_similar])],
            emotions[int(indices[1][most_similar])],
        ],
        "least_similar_pair": [
            emotions[int(indices[0][least_similar])],
            emotions[int(indices[1][least_similar])],
        ],
        "within_negative_cosine": _finite(
            _group_mean(cosine, emotions, NEGATIVE_EMOTIONS)
        ),
        "within_positive_cosine": _finite(
            _group_mean(cosine, emotions, POSITIVE_EMOTIONS)
        ),
        "across_valence_cosine": _finite(
            _group_mean(cosine, emotions, NEGATIVE_EMOTIONS, POSITIVE_EMOTIONS)
        ),
    }


def random_direction_cosine(
    count: int,
    hidden_size: int,
    *,
    seed: int,
    repeats: int = 20,
) -> dict:
    """Mean pairwise cosine of random unit directions, the no-structure reference."""
    generator = torch.Generator().manual_seed(seed)
    batches = [
        upper_triangle_values(
            cosine_matrix(torch.randn(count, hidden_size, generator=generator))
        )
        for _ in range(repeats)
    ]
    values = torch.cat(batches)
    return {
        "mean_pairwise_cosine": _finite(float(values.mean())),
        "mean_abs_pairwise_cosine": _finite(float(values.abs().mean())),
        "max_abs_pairwise_cosine": _finite(float(values.abs().max())),
        "repeats": repeats,
    }


# ------------------------------------------------------------------- the PCA


def pca(matrix: torch.Tensor, n_components: int | None = None) -> dict:
    """PCA of a small set of row vectors, centered across rows."""
    rows = matrix.float()
    row_count = rows.shape[0]
    centered = rows - rows.mean(dim=0, keepdim=True)
    _u, singular_values, basis = torch.linalg.svd(centered, full_matrices=False)
    variances = singular_values**2
    total = float(variances.sum())
    ratios = (variances / total) if total > 0 else torch.zeros_like(variances)
    available = min(row_count - 1, int(singular_values.numel()))
    keep = available if n_components is None else min(n_components, available)
    coordinates = centered @ basis[:keep].T
    return {
        "coordinates": coordinates,
        "components": basis[:keep],
        "singular_values": singular_values[:available].tolist(),
        "explained_variance_ratio": ratios[:available].tolist(),
        "component_count": keep,
    }


def axis_valence_separation(axis: torch.Tensor, emotions: Sequence[str]) -> dict:
    """Does one PCA axis split positive from negative emotions by sign?

    Sign is free in PCA, so both orientations are tried and the better one kept.
    """
    values = axis.float().tolist()
    labels = [1 if emotion in POSITIVE_EMOTIONS else -1 for emotion in emotions]
    best_accuracy = 0.0
    best_sign = 1
    for sign in (1, -1):
        correct = sum(
            1
            for value, label in zip(values, labels, strict=True)
            if (sign * value > 0) == (label > 0)
        )
        accuracy = correct / len(values)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_sign = sign
    oriented = [best_sign * value for value in values]
    positive = [
        value
        for value, label in zip(oriented, labels, strict=True)
        if label > 0
    ]
    negative = [
        value
        for value, label in zip(oriented, labels, strict=True)
        if label < 0
    ]
    return {
        "sign_accuracy": best_accuracy,
        "orientation": best_sign,
        "positive_group_mean": _finite(sum(positive) / len(positive))
        if positive
        else None,
        "negative_group_mean": _finite(sum(negative) / len(negative))
        if negative
        else None,
        "group_gap": _finite(
            (sum(positive) / len(positive)) - (sum(negative) / len(negative))
        )
        if positive and negative
        else None,
    }
