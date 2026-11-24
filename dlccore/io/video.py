"""Video frames through OpenCV, lifted from napari-deeplabcut's ``Video``: RGB ``uint8`` frames by index."""

from __future__ import annotations

from pathlib import Path
from types import TracebackType

import cv2
import numpy as np

_MAX_FORWARD_DECODE = 8


class Video:
    """Random access to a video's frames, keeping decoder state between reads.

    A forward jump of at most eight frames decodes forward instead of seeking.
    """

    def __init__(self, video_path: str | Path) -> None:
        """Open a video.

        Args:
            video_path: The video file.

        Raises:
            ValueError: If ``video_path`` is not a file.
            OSError: If OpenCV cannot open it.
        """

        if not Path(video_path).is_file():
            raise ValueError(f'Video path "{video_path}" does not point to a file.')

        self._path = str(video_path)
        self._stream = cv2.VideoCapture(self._path)
        if not self._stream.isOpened():
            raise OSError("Video could not be opened.")

        self._n_frames = int(self._stream.get(cv2.CAP_PROP_FRAME_COUNT))
        self._width = int(self._stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = float(self._stream.get(cv2.CAP_PROP_FPS))
        self._pos = 0  # next frame to read

    def __len__(self) -> int:
        """Return the frame count OpenCV reports."""

        return self._n_frames

    def __enter__(self) -> Video:
        """Return this video, to close on leaving the block."""

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release the stream."""

        self.close()

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def fps(self) -> float:
        """Frame rate the container reports."""

        return self._fps

    def _seek(self, ind: int) -> None:
        """Position the stream so the next read returns frame ``ind``, clamped to the video.

        Raises:
            OSError: If OpenCV refuses the seek.
        """

        ind = max(0, min(ind, len(self) - 1))
        if ind != self._pos:
            ok = self._stream.set(cv2.CAP_PROP_POS_FRAMES, ind)
            if not ok:
                raise OSError(f"Failed to seek to frame {ind}")
            self._pos = ind

    def _read_next(self) -> np.ndarray:
        """Decode the next frame as RGB.

        Raises:
            IndexError: If no frame could be read.
        """

        ok, frame = self._stream.read()
        if not ok:
            raise IndexError("Could not read frame")
        self._pos += 1
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _grab_next(self) -> bool:
        """Advance one frame without converting it.

        Raises:
            IndexError: If no frame could be grabbed.
        """

        ok = self._stream.grab()
        if not ok:
            raise IndexError("Could not grab frame")
        self._pos += 1
        return ok

    def read_frame(self, ind: int) -> np.ndarray:
        """Return frame ``ind`` as an RGB ``uint8`` array of shape ``(height, width, 3)``.

        Raises:
            IndexError: If ``ind`` is outside the video.
        """

        if ind < 0 or ind >= len(self):
            raise IndexError(f"Frame index {ind} out of bounds for video with {len(self)} frames.")

        if ind == self._pos:
            return self._read_next()

        if ind > self._pos and (ind - self._pos) <= _MAX_FORWARD_DECODE:
            while self._pos < ind:
                self._grab_next()
            return self._read_next()

        self._seek(ind)
        return self._read_next()

    def read_frames(self, start: int, stop: int) -> np.ndarray:
        """Return frames ``start`` to ``stop`` (exclusive), clamped to the video, as ``(n, height, width, 3)``."""

        start = max(0, start)
        stop = min(stop, len(self))
        if stop <= start:
            return np.empty((0, self.height, self.width, 3), dtype=np.uint8)

        self._seek(start)
        out = np.empty((stop - start, self.height, self.width, 3), dtype=np.uint8)

        for i in range(stop - start):
            out[i] = self._read_next()

        return out

    def close(self) -> None:
        """Release the stream."""

        self._stream.release()
