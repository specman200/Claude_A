"""The geometry that guarantees boxes land where the objects are."""

import numpy as np
import pytest

from ppe.letterbox import PAD_VALUE, fit, letterbox

SIZES = [(1920, 1080), (1280, 720), (640, 480), (480, 640), (700, 700), (1, 5)]


@pytest.mark.parametrize("w,h", SIZES)
@pytest.mark.parametrize("imgsz", [320, 640, 960])
def test_canvas_is_square_and_padded(w, h, imgsz):
    canvas, meta = letterbox(np.zeros((h, w, 3), np.uint8), imgsz)
    assert canvas.shape == (imgsz, imgsz, 3)
    assert meta.src_w == w and meta.src_h == h
    # Content is centred: padding on both sides of the short axis.
    assert 0 <= meta.pad_x <= imgsz and 0 <= meta.pad_y <= imgsz


@pytest.mark.parametrize("w,h", SIZES)
def test_aspect_ratio_is_preserved(w, h):
    _, meta = letterbox(np.zeros((h, w, 3), np.uint8), 640)
    scaled_w, scaled_h = w * meta.gain, h * meta.gain
    assert scaled_w == pytest.approx(640, abs=1) or scaled_h == pytest.approx(640, abs=1)
    assert scaled_w / scaled_h == pytest.approx(w / h, rel=1e-6)


@pytest.mark.parametrize("w,h", SIZES)
@pytest.mark.parametrize("imgsz", [320, 640])
def test_boxes_round_trip_to_source_pixels(w, h, imgsz):
    """A box drawn around known source pixels must come back to those pixels."""
    source = np.array(
        [[0, 0, w, h], [w * 0.25, h * 0.25, w * 0.75, h * 0.75], [w * 0.5, h * 0.5, w, h]],
        dtype=np.float32,
    )
    _, meta = letterbox(np.zeros((h, w, 3), np.uint8), imgsz)
    # Forward: source -> canvas, exactly as letterbox transformed the pixels.
    canvas = source * meta.gain
    canvas[:, [0, 2]] += meta.pad_x
    canvas[:, [1, 3]] += meta.pad_y

    assert meta.to_source(canvas) == pytest.approx(source, abs=1e-3)


def test_pixel_content_lands_where_the_transform_says():
    """A white square in the source must be white at the mapped canvas box."""
    frame = np.zeros((720, 1280, 3), np.uint8)
    frame[200:400, 500:900] = 255
    canvas, meta = letterbox(frame, 640)

    box = meta.to_source(
        np.array([[500 * meta.gain + meta.pad_x, 200 * meta.gain + meta.pad_y,
                   900 * meta.gain + meta.pad_x, 400 * meta.gain + meta.pad_y]], np.float32)
    )[0]
    assert box == pytest.approx([500, 200, 900, 400], abs=1.0)

    cx = int((500 + 900) / 2 * meta.gain + meta.pad_x)
    cy = int((200 + 400) / 2 * meta.gain + meta.pad_y)
    assert canvas[cy, cx].tolist() == [255, 255, 255]
    assert canvas[0, 0].tolist() == [PAD_VALUE] * 3  # letterbox bar, untouched


def test_boxes_are_clipped_to_the_frame():
    _, meta = letterbox(np.zeros((720, 1280, 3), np.uint8), 640)
    out = meta.to_source(np.array([[-50, -50, 1200, 1200]], np.float32))[0]
    assert out.tolist() == [0.0, 0.0, 1280.0, 720.0]


def test_empty_detections_stay_empty():
    _, meta = letterbox(np.zeros((480, 640, 3), np.uint8), 640)
    assert meta.to_source(np.empty((0, 4), np.float32)).shape == (0, 4)


def test_no_upscale_when_disabled():
    _, meta = letterbox(np.zeros((240, 320, 3), np.uint8), 640, scaleup=False)
    assert meta.gain == 1.0


@pytest.mark.parametrize(
    "src,dst,expect",
    [
        ((100, 100), (200, 200), (2.0, 0.0, 0.0)),
        ((200, 100), (200, 200), (1.0, 0.0, 50.0)),   # letterboxed top/bottom
        ((100, 200), (200, 200), (1.0, 50.0, 0.0)),   # pillarboxed left/right
    ],
)
def test_fit_centres_the_image_in_the_widget(src, dst, expect):
    assert fit(*src, *dst) == pytest.approx(expect)


def test_fit_survives_a_zero_sized_widget():
    assert fit(0, 0, 100, 100) == (1.0, 0.0, 0.0)
