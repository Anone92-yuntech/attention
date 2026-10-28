"""Action-window identity safety and source-time output, without any models."""
import csv
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attention_log import ActionWindowLog


def frame(playing, target_id=1, generation=1):
    return SimpleNamespace(playing=playing, tracking=SimpleNamespace(
        target_id=target_id, selection_generation=generation))


class ActionWindowLogTests(unittest.TestCase):
    def logger(self, fps=1, start_seconds=0):
        stream = io.StringIO(newline="")
        return ActionWindowLog(stream, fps, start_seconds), stream

    def test_three_positive_frames_trigger_but_two_do_not(self):
        for positives, expected in ((2, "none"), (3, "playing")):
            with self.subTest(positives=positives):
                log, _ = self.logger()
                for index in range(5):
                    log.add(frame(index < positives))
                self.assertEqual(log.rows[0]["Status"], expected)
                self.assertEqual(log.rows[0]["Trigger_Frames"], positives)
                self.assertFalse(log.rows[0]["Partial_Window"])

    def test_unknown_is_not_none(self):
        for inputs in ([None] * 5, [False, False, False, False, None]):
            log, _ = self.logger()
            for value in inputs:
                log.add(frame(value))
            self.assertEqual(log.rows[0]["Status"], "unknown")

    def test_positive_evidence_can_trigger_with_same_identity_unknown_frames(self):
        log, _ = self.logger()
        for value in (None, True, True, True, None):
            log.add(frame(value))
        self.assertEqual(log.rows[0]["Status"], "playing")
        self.assertEqual(log.rows[0]["Unknown_Frames"], 2)

    def test_initialization_none_to_first_id_is_not_mixed(self):
        log, _ = self.logger()
        log.add(frame(None, target_id=None))
        for value in (True, True, True, False):
            log.add(frame(value))
        self.assertEqual(log.rows[0]["Status"], "playing")
        self.assertFalse(log.rows[0]["Mixed_Identity"])
        self.assertEqual(log.rows[0]["Target_ID"], 1)

    def test_cross_id_positives_do_not_combine(self):
        log, _ = self.logger()
        for target_id, value in ((1, True), (1, True), (2, True), (2, False), (2, False)):
            log.add(frame(value, target_id=target_id))
        row = log.rows[0]
        self.assertEqual(row["Status"], "unknown")
        self.assertTrue(row["Mixed_Identity"])
        self.assertIsNone(row["Target_ID"])
        self.assertEqual(row["Trigger_Frames"], 1)
        self.assertEqual(row["Discarded_Frames"], 2)
        self.assertEqual(row["Valid_Frames"] + row["Unknown_Frames"] + row["Discarded_Frames"], 5)

    def test_same_id_new_generation_does_not_combine(self):
        log, _ = self.logger()
        for generation, value in ((1, True), (1, True), (2, True), (2, False), (2, False)):
            log.add(frame(value, generation=generation))
        row = log.rows[0]
        self.assertEqual(row["Status"], "unknown")
        self.assertTrue(row["Mixed_Identity"])
        self.assertIsNone(row["Selection_Generation"])
        self.assertEqual(row["Trigger_Frames"], 1)
        self.assertEqual(row["Discarded_Frames"], 2)

    def test_generation_change_before_lock_still_marks_mixed(self):
        log, _ = self.logger()
        log.add(frame(None, target_id=None, generation=1))
        for _ in range(4):
            log.add(frame(True, generation=2))
        self.assertEqual(log.rows[0]["Status"], "unknown")
        self.assertTrue(log.rows[0]["Mixed_Identity"])
        self.assertEqual(log.rows[0]["Discarded_Frames"], 1)

    def test_retained_id_loss_is_unknown_but_not_mixed(self):
        log, _ = self.logger()
        for value in (False, False, None, False, False):
            log.add(frame(value))
        self.assertEqual(log.rows[0]["Status"], "unknown")
        self.assertFalse(log.rows[0]["Mixed_Identity"])

    def test_source_start_time_full_window_then_partial(self):
        log, stream = self.logger(fps=2, start_seconds=8)
        for _ in range(13):
            log.add(frame(False))
        self.assertEqual(log.rows[0]["Time_Window"], "08s - 13s")
        log.finish()
        self.assertEqual(log.rows[1]["Time_Window"], "13s - 14.5s")
        self.assertTrue(log.rows[1]["Partial_Window"])
        self.assertEqual(log.rows[1]["Valid_Frames"], 3)
        self.assertFalse(stream.closed)
        decoded = list(csv.DictReader(io.StringIO(stream.getvalue())))
        self.assertEqual(len(decoded), 2)
        self.assertEqual(list(decoded[0])[:2], ["Time_Window", "Status"])
        json.dumps(log.rows)

    def test_finish_is_idempotent_and_exact_window_has_no_extra_row(self):
        log, stream = self.logger()
        for _ in range(5):
            log.add(frame(False))
        log.finish()
        before = stream.getvalue()
        log.finish()
        self.assertEqual(stream.getvalue(), before)
        self.assertEqual(len(log.rows), 1)
        with self.assertRaises(RuntimeError):
            log.add(frame(False))

    def test_finish_empty_log_writes_no_data_row(self):
        log, stream = self.logger()
        log.finish()
        log.finish()
        self.assertEqual(log.rows, [])
        self.assertEqual(len(stream.getvalue().splitlines()), 1)

    def test_new_identity_at_window_boundary_is_separate_and_valid(self):
        log, _ = self.logger()
        for target_id, generation in ((1, 1), (2, 2)):
            for _ in range(5):
                log.add(frame(True, target_id, generation))
        self.assertEqual([row["Status"] for row in log.rows], ["playing", "playing"])
        self.assertEqual([row["Target_ID"] for row in log.rows], [1, 2])
        self.assertFalse(any(row["Mixed_Identity"] for row in log.rows))

    def test_invalid_parameters_rejected(self):
        for fps in (0, -1, float("nan"), float("inf")):
            with self.subTest(fps=fps), self.assertRaises(ValueError):
                self.logger(fps=fps)
        for start in (-1, float("nan"), float("inf")):
            with self.subTest(start=start), self.assertRaises(ValueError):
                self.logger(start_seconds=start)


if __name__ == "__main__":
    unittest.main()
