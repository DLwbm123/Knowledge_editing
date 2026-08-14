import importlib.util
import sys
from pathlib import Path

import numpy as np
from PIL import Image


LLAVA_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "LLaVA-Med"
MODULE_PATH = LLAVA_ROOT / "llava" / "mm_utils.py"
if str(LLAVA_ROOT) not in sys.path:
    sys.path.insert(0, str(LLAVA_ROOT))
spec = importlib.util.spec_from_file_location("llava_med_mm_utils_determinism_test", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _nonzero_bounds(image):
    array = np.asarray(image)
    mask = array.sum(axis=-1) > 0
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def test_expand2square_wide_image_is_centered_and_repeatable():
    image = Image.new("RGB", (7, 4), (255, 255, 255))
    outputs = [module.expand2square(image, (0, 0, 0)) for _ in range(8)]
    assert all(np.array_equal(np.asarray(outputs[0]), np.asarray(item)) for item in outputs[1:])
    assert outputs[0].size == (7, 7)
    assert _nonzero_bounds(outputs[0]) == (0, 1, 6, 4)


def test_expand2square_tall_image_is_centered_and_repeatable():
    image = Image.new("RGB", (4, 7), (255, 255, 255))
    outputs = [module.expand2square(image, (0, 0, 0)) for _ in range(8)]
    assert all(np.array_equal(np.asarray(outputs[0]), np.asarray(item)) for item in outputs[1:])
    assert outputs[0].size == (7, 7)
    assert _nonzero_bounds(outputs[0]) == (1, 0, 4, 6)
