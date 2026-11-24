#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/dlccore
#
# Contributors:
# https://github.com/DeepLabCut/dlccore/graphs/contributors
#
# Licensed under GNU Lesser General Public License v3.0
#

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="OpenCV comes with the optional 'video' extra")

from dlccore.io.video import Video  # noqa: E402

if TYPE_CHECKING:
    from deeplabcut.utils.auxfun_videos import VideoReader

N_FRAMES = 60
WIDTH, HEIGHT = 128, 96
FPS = 10
BASE, LEVEL_OFFSET, LEVEL_STEP = 8, 16, 32
DIGIT = 32  # side of each digit's square, in pixels

# First, small forward jump, large forward jump to the last, backward, sequential, small forward, large forward.
# With OpenCV's mp4v writer, keyframes fall every 12 frames; 17, 50 and 59 are inter frames.
ACCESS_ORDER = [0, 5, N_FRAMES - 1, 17, 18, 25, 50]

CODECS = {"MJPG": ".avi", "mp4v": ".mp4"}

# A fixed textured background keeps most of each frame static, so mp4v writes inter frames rather than all keyframes.
_BACKGROUND = cv2.GaussianBlur(np.random.default_rng(0).integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8), (9, 9), 3)


def _encoded(index: int) -> np.ndarray:
    """Frame ``index``: its two base-8 digits as flat grey squares, levels far apart to survive lossy codecs."""

    frame = _BACKGROUND.copy()
    frame[:DIGIT, :DIGIT] = LEVEL_OFFSET + LEVEL_STEP * (index // BASE)
    frame[:DIGIT, DIGIT : 2 * DIGIT] = LEVEL_OFFSET + LEVEL_STEP * (index % BASE)
    return frame


def _index_of(frame: np.ndarray) -> int:
    """Map a decoded frame back to the index it was written at, reading each square away from its edges."""

    def digit(square: np.ndarray) -> int:
        return round((float(square.mean()) - LEVEL_OFFSET) / LEVEL_STEP)

    inner = slice(8, DIGIT - 8)
    high = frame[inner, inner]
    low = frame[inner, DIGIT + 8 : 2 * DIGIT - 8]
    return BASE * digit(high) + digit(low)


def _write_video(path: Path, fourcc: str) -> bool:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        return False
    for index in range(N_FRAMES):
        writer.write(_encoded(index))
    writer.release()

    capture = cv2.VideoCapture(str(path))
    decoded = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        decoded.append(_index_of(frame))
    capture.release()
    return decoded == list(range(N_FRAMES))


@pytest.fixture(scope="module", params=list(CODECS))
def video_path(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Path:
    fourcc = request.param
    path = tmp_path_factory.mktemp(f"video_{fourcc}") / f"indexed{CODECS[fourcc]}"
    if not _write_video(path, fourcc):
        pytest.skip(f"This OpenCV build cannot write and decode back a {fourcc} video.")
    return path


def test_the_fixture_encodes_each_index_distinctly() -> None:
    # Control: without compression, every index round-trips.
    assert [_index_of(_encoded(index)) for index in range(N_FRAMES)] == list(range(N_FRAMES))


def test_the_mp4v_fixture_has_inter_frames_at_the_accessed_indices(video_path: Path) -> None:
    # Control: otherwise the mp4v equivalence tests would only ever land on keyframes.
    if video_path.suffix != ".mp4":
        pytest.skip("Only the mp4v fixture is meant to have inter frames.")
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        pytest.skip("ffprobe is not on PATH.")
    arguments = ["-v", "error", "-select_streams", "v:0", "-show_entries", "frame=pict_type", "-of", "csv=p=0"]
    probed = subprocess.run(
        [ffprobe, *arguments, str(video_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    picture_types = [line.strip().strip(",") for line in probed.stdout.splitlines() if line.strip()]

    assert len(picture_types) == N_FRAMES
    assert [index for index in ACCESS_ORDER if picture_types[index] == "I"] == [0]


def test_video_properties(video_path: Path) -> None:
    video = Video(video_path)

    assert (len(video), video.width, video.height, video.fps) == (N_FRAMES, WIDTH, HEIGHT, FPS)

    video.close()
    assert not video._stream.isOpened()


def test_a_context_manager_closes_the_video(video_path: Path) -> None:
    with Video(video_path) as video:
        assert video._stream.isOpened()

    assert not video._stream.isOpened()


def test_a_frame_is_rgb_uint8_of_the_video_shape(video_path: Path) -> None:
    with Video(video_path) as video:
        frame = video.read_frame(0)

    assert (frame.shape, frame.dtype) == ((HEIGHT, WIDTH, 3), np.uint8)


def test_frames_are_rgb(tmp_path: Path) -> None:
    path = tmp_path / "blue.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        pytest.skip("This OpenCV build cannot write MJPG.")
    bgr_blue = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    bgr_blue[..., 0] = 255
    writer.write(bgr_blue)
    writer.release()

    with Video(path) as video:
        red, _green, blue = video.read_frame(0).reshape(-1, 3).mean(axis=0)

    assert (blue > 200, red < 50) == (True, True)


@pytest.mark.parametrize("index", [N_FRAMES, -1])
def test_a_frame_outside_the_video_is_an_index_error(video_path: Path, index: int) -> None:
    with Video(video_path) as video, pytest.raises(IndexError):
        video.read_frame(index)


def test_sequential_reads_return_consecutive_frames(video_path: Path) -> None:
    with Video(video_path) as video:
        indices = [_index_of(video.read_frame(index)) for index in range(N_FRAMES)]

    assert indices == list(range(N_FRAMES))


def test_a_block_read_returns_the_frames_in_range(video_path: Path) -> None:
    with Video(video_path) as video:
        video.read_frame(30)
        block = video.read_frames(10, 20)

    assert (block.shape, block.dtype) == ((10, HEIGHT, WIDTH, 3), np.uint8)
    assert [_index_of(frame) for frame in block] == list(range(10, 20))


def test_a_block_read_is_clamped_to_the_video(video_path: Path) -> None:
    with Video(video_path) as video:
        assert video.read_frames(N_FRAMES - 2, N_FRAMES + 5).shape[0] == 2
        assert video.read_frames(5, 5).shape == (0, HEIGHT, WIDTH, 3)


def test_a_path_that_is_not_a_file_is_a_value_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not point to a file"):
        Video(tmp_path)


def test_a_file_opencv_cannot_open_is_an_os_error(tmp_path: Path) -> None:
    path = tmp_path / "not_a_video.mp4"
    path.write_bytes(b"not a video")

    with pytest.raises(OSError, match="could not be opened"):
        Video(path)


class _Refusing:
    """The real stream, with one decoder call failing as OpenCV reports it: a false ``ok``."""

    def __init__(self, stream: cv2.VideoCapture, refused: str) -> None:
        self._stream = stream
        self._refused = refused

    def set(self, *args: object) -> bool:
        return False if self._refused == "set" else self._stream.set(*args)

    def read(self) -> tuple[bool, np.ndarray | None]:
        return (False, None) if self._refused == "read" else self._stream.read()

    def grab(self) -> bool:
        return False if self._refused == "grab" else self._stream.grab()

    def release(self) -> None:
        self._stream.release()


@pytest.mark.parametrize(
    ("refused", "index", "error", "message"),
    [
        ("set", N_FRAMES - 1, OSError, "seek to frame"),
        ("read", 0, IndexError, "read frame"),
        ("grab", 3, IndexError, "grab frame"),
    ],
    ids=["seek", "read", "grab"],
)
def test_a_decoder_failure_is_raised_not_returned_as_a_frame(
    video_path: Path, refused: str, index: int, error: type[Exception], message: str
) -> None:
    with Video(video_path) as video:
        video._stream = _Refusing(video._stream, refused)

        with pytest.raises(error, match=message):
            video.read_frame(index)


def test_read_frame_decodes_the_requested_frame(video_path: Path) -> None:
    with Video(video_path) as video:
        decoded = [_index_of(video.read_frame(index)) for index in ACCESS_ORDER]

    assert decoded == ACCESS_ORDER


def test_read_frame_equals_deeplabcut_opencv_extraction(video_path: Path) -> None:
    # DeepLabCut's extract_frames(opencv=True): set_to_frame, then read_frame(crop=True); it returns RGB.
    auxfun_videos = pytest.importorskip("deeplabcut.utils.auxfun_videos")
    reader = auxfun_videos.VideoReader(str(video_path))
    try:
        expected = [_extract_with_deeplabcut(reader, index) for index in ACCESS_ORDER]
    finally:
        reader.close()

    with Video(video_path) as video:
        for index, frame in zip(ACCESS_ORDER, expected, strict=True):
            np.testing.assert_array_equal(video.read_frame(index), frame, err_msg=f"frame {index}")


def test_read_frame_equals_moviepy_extraction(video_path: Path) -> None:
    # DeepLabCut's extract_frames(opencv=False): VideoFileClip.get_frame(index / fps), moviepy 1.x.
    editor = pytest.importorskip("moviepy.editor")

    clip = editor.VideoFileClip(str(video_path))
    try:
        expected = [_index_of(clip.get_frame(index * 1.0 / clip.fps)) for index in ACCESS_ORDER]
    finally:
        clip.close()

    with Video(video_path) as video:
        decoded = [_index_of(video.read_frame(index)) for index in ACCESS_ORDER]

    assert decoded == expected == ACCESS_ORDER


def _extract_with_deeplabcut(reader: VideoReader, index: int) -> np.ndarray:
    reader.set_to_frame(index)
    frame = reader.read_frame(crop=True)
    assert frame is not None, f"DeepLabCut read no frame at {index}"
    return frame
