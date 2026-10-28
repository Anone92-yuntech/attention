"""Five-second action windows with explicit unknown and identity boundaries."""
from __future__ import annotations

import csv
from math import isclose, isfinite


FIELDS = (
    "Time_Window", "Status", "Target_ID", "Selection_Generation",
    "Trigger_Frames", "Valid_Frames", "Unknown_Frames", "Mixed_Identity",
    "Partial_Window", "Discarded_Frames",
)


def _seconds(value: float) -> str:
    if isclose(value, round(value), abs_tol=1e-7):
        return f"{round(value):02d}s"
    return f"{value:.3f}".rstrip("0").rstrip(".") + "s"


class ActionWindowLog:
    """Write one row per five seconds without combining different people.

    ``Valid_Frames``, ``Unknown_Frames`` and ``Trigger_Frames`` describe only
    the segment after the most recent identity switch in a window. Earlier
    samples are counted in ``Discarded_Frames`` and cannot trigger an action.
    Any window spanning identities/generations is explicitly ``unknown``.
    The supplied stream remains owned by the caller.
    """

    def __init__(self, stream, fps: float, start_seconds: float = 0.0):
        self.fps = float(fps)
        self.start_seconds = float(start_seconds)
        if not isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be finite and positive")
        if not isfinite(self.start_seconds) or self.start_seconds < 0:
            raise ValueError("start_seconds must be finite and nonnegative")
        self.window_frames = max(1, round(self.fps * 5))
        self.stream = stream
        self.writer = csv.DictWriter(stream, fieldnames=FIELDS)
        self.rows: list[dict] = []
        self._processed = 0
        self._finished = False
        self._clear_window()
        self.writer.writeheader()
        self.stream.flush()

    def _clear_window(self):
        self._window_start = self._processed
        self._count = 0
        self._valid = 0
        self._unknown = 0
        self._triggers = 0
        self._discarded = 0
        self._ids = set()
        self._generations = set()
        self._last_generation = None
        self._have_generation = False
        self._last_known_id = None

    def add(self, attention_frame):
        if self._finished:
            raise RuntimeError("Cannot add frames after finish()")
        tracking = attention_frame.tracking
        target_id = tracking.target_id
        generation = tracking.selection_generation
        generation_changed = self._have_generation and generation != self._last_generation
        target_changed = (target_id is not None and self._last_known_id is not None
                          and target_id != self._last_known_id)
        if generation_changed or target_changed:
            self._discarded += self._valid + self._unknown
            self._valid = self._unknown = self._triggers = 0
            if generation_changed:
                self._last_known_id = None
        self._last_generation = generation
        self._have_generation = True
        if target_id is not None:
            self._last_known_id = target_id
            self._ids.add(target_id)
        self._generations.add(generation)

        playing = attention_frame.playing if target_id is not None else None
        if playing is None:
            self._unknown += 1
        else:
            self._valid += 1
            self._triggers += int(bool(playing))
        self._count += 1
        self._processed += 1
        if self._count == self.window_frames:
            self._emit(partial=False)

    def _emit(self, *, partial: bool):
        mixed = len(self._ids) > 1 or len(self._generations) > 1
        if mixed or self._valid == 0:
            status = "unknown"
        elif self._triggers >= 3:
            status = "playing"
        else:
            status = "unknown" if self._unknown else "none"
        start = self.start_seconds + self._window_start / self.fps
        end = self.start_seconds + self._processed / self.fps
        row = {
            "Time_Window": f"{_seconds(start)} - {_seconds(end)}",
            "Status": status,
            "Target_ID": next(iter(self._ids)) if len(self._ids) == 1 and not mixed else None,
            "Selection_Generation": next(iter(self._generations)) if len(self._generations) == 1 else None,
            "Trigger_Frames": self._triggers,
            "Valid_Frames": self._valid,
            "Unknown_Frames": self._unknown,
            "Mixed_Identity": mixed,
            "Partial_Window": partial,
            "Discarded_Frames": self._discarded,
        }
        self.writer.writerow(row)
        self.stream.flush()
        self.rows.append(row)
        self._clear_window()

    def finish(self):
        """Flush an incomplete final window once; never close the stream."""
        if not self._finished:
            if self._count:
                self._emit(partial=True)
            self._finished = True
        return self.rows
