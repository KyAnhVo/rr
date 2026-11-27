"""Classes shared by stages 4-5 and the evaluation.

Stage 5 votes over a few coarse classes. All but the last are the room's *structure*; the
last, "object", is everything movable. Unknown is -1.
"""
import numpy as np
from numpy.typing import NDArray

CLASSES = ("wall", "floor", "ceiling", "door", "window", "other structure", "object")
OBJECT = len(CLASSES) - 1
UNKNOWN = -1

# RGB. Structure in cool/earthy tones, objects in red, so the split reads at a glance.
COLORS = np.array([
    (174, 199, 232),  # wall
    (152, 223, 138),  # floor
    (255, 187, 120),  # ceiling
    (140, 86, 75),    # door
    (148, 103, 189),  # window
    (23, 190, 207),   # other structure
    (214, 39, 40),    # object
], np.uint8)
UNKNOWN_COLOR = np.array((150, 150, 150), np.uint8)
STRUCTURE_COLOR = np.array((31, 119, 180), np.uint8)
OBJECT_COLOR = np.array((255, 127, 14), np.uint8)

# ADE20K names (as in the model's id2label; synonyms separated by commas) -> coarse class.
_ADE = {
    "wall": 0,
    "floor": 1, "flooring": 1,
    "ceiling": 2,
    "door": 3, "double door": 3, "screen door": 3,
    "windowpane": 4, "window": 4, "skylight": 4,
    "column": 5, "pillar": 5, "stairs": 5, "steps": 5, "stairway": 5, "staircase": 5, "step": 5,
    "stair": 5, "railing": 5, "rail": 5, "bannister": 5, "banister": 5, "balustrade": 5, "handrail": 5,
}

# Replica semantic-mesh class names -> coarse class. Fixed fixtures count as structure.
_REPLICA = {
    "wall": 0, "floor": 1, "ceiling": 2, "door": 3, "window": 4,
    "beam": 5, "pillar": 5, "stair": 5, "handrail": 5, "vent": 5, "switch": 5, "wall-plug": 5,
    "pipe": 5, "panel": 5,
}
_REPLICA_IGNORE = ("undefined", "anonymize")


def ade_class(name: str) -> int:
    synonyms = [s.strip().lower() for s in name.replace(";", ",").split(",")]
    return next((_ADE[s] for s in synonyms if s in _ADE), OBJECT)


def ade_lookup(id2label: dict[int, str]) -> NDArray[np.int8]:
    """Table from a stored label value (0-255) to coarse class; values without a name are unknown."""
    table = np.full(256, UNKNOWN, np.int8)
    for i, name in id2label.items():
        table[int(i)] = ade_class(name)
    return table


def replica_class(name: str) -> int:
    """Coarse class of a Replica class name; unknown for unlabelled or anonymised surfaces."""
    name = name.strip().lower()
    if not name or name.startswith(_REPLICA_IGNORE):
        return UNKNOWN
    return _REPLICA.get(name, OBJECT)


def is_structure(classes: NDArray) -> NDArray[np.bool_]:
    return (classes >= 0) & (classes < OBJECT)


def class_colors(classes: NDArray) -> NDArray[np.uint8]:
    """RGB per coarse class."""
    colors = np.tile(UNKNOWN_COLOR, (len(classes), 1))
    known = classes >= 0
    colors[known] = COLORS[classes[known]]
    return colors


def binary_colors(classes: NDArray) -> NDArray[np.uint8]:
    """RGB: structure blue, object orange, unknown grey."""
    colors = np.tile(UNKNOWN_COLOR, (len(classes), 1))
    colors[is_structure(classes)] = STRUCTURE_COLOR
    colors[classes == OBJECT] = OBJECT_COLOR
    return colors
