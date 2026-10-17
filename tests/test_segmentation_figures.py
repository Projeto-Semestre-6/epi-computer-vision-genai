from pathlib import Path

import numpy as np
import pytest

from conftest import load_script

figures = load_script("make_segmentation_figures")
IDENTIFIED, ABSENT, INDETERMINATE = "identificado", "ausência identificada", "indeterminado"


def states_of(detections):
    return {ppe: info["estado"] for ppe, info in figures.ppe_states(detections).items()}


def test_identified_absent_and_not_detected():
    assert states_of([("helmet", 0.9), ("no-vest", 0.4)]) == {
        "Capacete": IDENTIFIED, "Colete": ABSENT, "Óculos": INDETERMINATE,
    }  # fmt: skip
    states = figures.ppe_states([("helmet", 0.9)])
    assert states["Óculos"]["motivo"] == "nenhuma detecção"
    assert states["Capacete"]["helmet"] == [0.9] and states["Capacete"]["no-helmet"] == []


def test_not_detected_is_never_absent():
    assert set(states_of([]).values()) == {INDETERMINATE}


@pytest.mark.parametrize("positive, negative", [("helmet", "no-helmet"), ("vest", "no-vest"), ("goggles", "no-goggles")])
def test_conflict_is_indeterminate(positive, negative):
    """Decision 1: the positive and the negative of one PPE in the same photo give "indeterminado"."""
    states = figures.ppe_states([(negative, 0.95), (positive, 0.3), (positive, 0.6)])
    ppe = next(name for name, pair in figures.PPE.items() if pair == (positive, negative))
    assert states[ppe]["estado"] == INDETERMINATE
    assert states[ppe]["motivo"] == "conflito"
    assert states[ppe][positive] == [0.6, 0.3] and states[ppe][negative] == [0.95]
    assert figures.state_text(states[ppe]) == "indeterminado (conflito)"


def test_confidence_threshold_and_out_of_scope_classes():
    assert states_of([("helmet", 0.2499), ("no-helmet", 0.25)])["Capacete"] == ABSENT
    assert states_of([("helmet", 0.1)])["Capacete"] == INDETERMINATE
    assert states_of([("boots", 0.99), ("gloves", 0.9), ("no-gloves", 0.9)]) == states_of([])
    assert states_of([("helmet", 0.3)]) == states_of([("helmet", 0.3), ("no-helmet", 0.2)])
    assert figures.discarded_classes([("helmet", 0.9), ("boots", 0.5), ("boots", 0.3), ("gloves", 0.4)]) == {
        "boots": 2, "gloves": 1,
    }  # fmt: skip


def test_truth_label():
    labels = figures.ppe_states([("helmet", 1.0), ("no-helmet", 1.0), ("no-vest", 1.0)])
    assert figures.truth_label(labels["Capacete"]) == "as duas classes anotadas"
    assert figures.truth_label(labels["Colete"]) == ABSENT
    assert figures.truth_label(labels["Óculos"]) == "não anotado"


def test_helmet_absence_candidates_in_alphabetical_order():
    truth = {"c.jpg": [("no-helmet", 1.0)], "a.jpg": [("no-helmet", 1.0)], "b.jpg": [("helmet", 1.0)],
             "d.jpg": [("no-helmet", 1.0), ("helmet", 1.0)], "e.jpg": [("no-helmet", 1.0)]}  # fmt: skip
    model = {"a.jpg": [("helmet", 0.5)], "c.jpg": [("no-helmet", 0.3)], "e.jpg": [("no-helmet", 0.8)]}
    asked = []

    def model_states(path: Path):
        asked.append(path.name)
        return figures.ppe_states(model.get(path.name, []))

    found = figures.helmet_absence_candidates(
        [Path(name) for name in ("e.jpg", "d.jpg", "c.jpg", "b.jpg", "a.jpg")],
        lambda path: figures.ppe_states(truth[path.name]),
        model_states,
    )
    assert next(found).name == "c.jpg"
    assert asked == ["a.jpg", "c.jpg"]  # the model only runs on images whose labels qualify
    assert [path.name for path in found] == ["e.jpg"]


def test_draw_figure_layout():
    pixels = np.zeros((40, 60, 3), np.uint8)
    states = figures.ppe_states([("helmet", 0.7), ("no-helmet", 0.3), ("vest", 0.9)])
    truth = figures.ppe_states([("helmet", 1.0)])
    boxes = [{"classe": "helmet", "conf": 0.7, "xyxy": [5, 5, 20, 20]}]
    figure = figures.draw_figure(pixels, boxes, states, truth, {"boots": 1})
    assert figure.size == (2 * 60 + 10, 26 + 40 + 26 + 22 * 5 + 10)
    left = np.asarray(figure)[26 : 26 + 40, :60]
    assert not left.any()  # the "before" panel is the input, untouched


def test_large_photo_is_shrunk_for_display_only():
    pixels = np.zeros((800, 1280, 3), np.uint8)
    states = figures.ppe_states([("no-helmet", 0.8)])
    boxes = [{"classe": "no-helmet", "conf": 0.8, "xyxy": [100, 100, 300, 300]}]
    figure = figures.draw_figure(pixels, boxes, states, states, {})
    assert figure.size == (2 * 640 + 10, 26 + 400 + 26 + 22 * 5 + 10)
    right = np.asarray(figure)[26 : 26 + 400, 650:]
    assert tuple(right[100, 50]) == figures.NEGATIVE_COLOR  # box edge at x 50, y 50..150 after halving
