import metrics


def test_perfect_prediction():
    y = [0, 0, 1, 1]
    m = metrics.compute_metrics(y, y, num_classes=2)
    assert m["macro_f1"] == 1.0
    assert m["balanced_acc"] == 1.0
    assert m["mcc"] == 1.0


def test_all_majority_is_penalized_by_macro_f1():
    y_true = [0, 0, 0, 1]      # imbalanced
    y_pred = [0, 0, 0, 0]      # predict majority always
    m = metrics.compute_metrics(y_true, y_pred, num_classes=2)
    # accuracy would be 0.75 but macro-F1 must be far lower
    assert m["macro_f1"] < 0.5
    assert m["balanced_acc"] == 0.5


def test_returns_per_class_lists():
    m = metrics.compute_metrics([0, 1], [0, 1], num_classes=2)
    assert len(m["per_class_precision"]) == 2
    assert len(m["per_class_recall"]) == 2
