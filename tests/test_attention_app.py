"""Runner, UI and failure-report checks with all neural models mocked."""
from contextlib import ExitStack, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np

import attention_app as app


class Capture:
    def __init__(self, frames=3, fps=2.0):
        self.limit = frames
        self.fps = fps
        self.read_count = 0
        self.position = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, key):
        return {
            app.cv2.CAP_PROP_FPS: self.fps,
            app.cv2.CAP_PROP_FRAME_WIDTH: 64,
            app.cv2.CAP_PROP_FRAME_HEIGHT: 48,
            app.cv2.CAP_PROP_POS_FRAMES: self.position,
            app.cv2.CAP_PROP_FRAME_COUNT: 100,
        }.get(key, 0)

    def set(self, key, value):
        if key == app.cv2.CAP_PROP_POS_MSEC:
            self.position = round(value * self.fps / 1000)
            return True
        return False

    def read(self):
        if self.read_count >= self.limit:
            return False, None
        self.read_count += 1
        self.position += 1
        return True, np.zeros((48, 64, 3), dtype=np.uint8)

    def release(self):
        self.released = True


class Analyzer:
    def __init__(self, *, fail_at=None, close_error=None, playing=False):
        self.fail_at = fail_at
        self.close_error = close_error
        self.playing = playing
        self.timestamps = []
        self.resets = 0
        self.closed = False

    def process(self, frame, timestamp):
        if len(self.timestamps) == self.fail_at:
            raise RuntimeError("mock model inference failed")
        self.timestamps.append(timestamp)
        return NS(
            tracking=NS(target_id=1, selection_generation=self.resets + 1,
                        state="locked", statistics={}),
            hand_boxes=[], toy_boxes=[], playing=self.playing,
            status="unknown" if self.playing is None else ("playing" if self.playing else "none"),
            detail="mock observation",
        )

    def reset_selection(self):
        self.resets += 1

    def close(self):
        self.closed = True
        if self.close_error:
            raise RuntimeError(self.close_error)


class VideoWriter:
    def __init__(self):
        self.frames = 0
        self.released = False

    def isOpened(self):
        return True

    def write(self, _frame):
        self.frames += 1

    def release(self):
        self.released = True


class AttentionAppTests(unittest.TestCase):
    def setUp(self):
        temporary_root = app.ROOT / ".cache" / "test_attention_app"
        temporary_root.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=temporary_root)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.input = self.root / "source.mp4"
        self.model = self.root / "toy.pt"
        self.input.touch()
        self.model.touch()
        self.output = self.root / "action.csv"
        self.report_path = self.root / "report.json"

    def args(self, *extra, headless=True):
        argv = ["--input", str(self.input), "--model", str(self.model),
                "--output-csv", str(self.output), "--report", str(self.report_path)]
        if headless:
            argv.append("--headless")
        return app.build_parser().parse_args(argv + list(extra))

    def dependencies(self, *, capture=None, analyzer=None, keys=None):
        capture = capture or Capture()
        analyzer = analyzer or Analyzer()
        writer = VideoWriter()
        stack = ExitStack()
        self.addCleanup(stack.close)
        mocks = {}
        replacements = {
            "PersonDetector": Mock(return_value=NS(provenance={"name": "mock detector"})),
            "YOLO": Mock(return_value=NS(device="cuda:0")),
            "TrackingPipeline": Mock(return_value=NS(close=Mock())),
            "AttentionAnalyzer": Mock(return_value=analyzer),
            "ensure_model": Mock(return_value=self.root / "pose.task"),
            "draw_analysis": Mock(side_effect=lambda frame, _result: frame.copy()),
        }
        for name, replacement in replacements.items():
            mocks[name] = stack.enter_context(patch.object(app, name, replacement))
        for name, replacement in {
            "VideoCapture": Mock(return_value=capture),
            "VideoWriter": Mock(return_value=writer),
            "namedWindow": Mock(), "resizeWindow": Mock(),
            "imshow": Mock(), "destroyAllWindows": Mock(),
            "waitKey": Mock(side_effect=keys) if keys is not None else Mock(return_value=-1),
            "getWindowProperty": Mock(return_value=1),
        }.items():
            mocks[name] = stack.enter_context(patch.object(app.cv2, name, replacement))
        return capture, analyzer, writer, mocks

    def test_headless_run_never_opens_gui_and_releases_models_and_video(self):
        capture, analyzer, _, mocks = self.dependencies()
        report = app.run(self.args())
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["end_reason"], "eof")
        self.assertEqual(analyzer.timestamps, [0, 500, 1000])
        self.assertTrue(capture.released)
        self.assertTrue(analyzer.closed)
        self.assertTrue(report["csv_rows"][0]["Partial_Window"])
        for name in ("namedWindow", "resizeWindow", "imshow", "waitKey", "draw_analysis"):
            mocks[name].assert_not_called()
        config = mocks["TrackingPipeline"].call_args.args[3]
        self.assertFalse(config.require_posture_landmarks)

    def test_headless_export_still_draws_and_writes_each_processed_frame(self):
        capture, analyzer, writer, mocks = self.dependencies()
        report = app.run(self.args("--export-video", str(self.root / "demo.mp4")))
        self.assertEqual(writer.frames, 3)
        self.assertEqual(mocks["draw_analysis"].call_count, 3)
        self.assertTrue(writer.released)
        self.assertTrue(capture.released and analyzer.closed)
        self.assertEqual(report["status"], "passed")
        mocks["imshow"].assert_not_called()

    def test_pause_does_not_read_frames_or_advance_source_time(self):
        capture, analyzer, _, mocks = self.dependencies(keys=[ord(" "), -1, -1, ord(" "), ord("q")])
        report = app.run(self.args(headless=False))
        self.assertEqual(capture.read_count, 2)
        self.assertEqual(analyzer.timestamps, [0, 500])
        self.assertEqual(report["end_reason"], "user_quit")
        self.assertEqual(mocks["waitKey"].call_count, 5)
        mocks["destroyAllWindows"].assert_called_once()

    def test_manual_reselection_reaches_analyzer_and_separates_csv_generations(self):
        _, analyzer, _, _ = self.dependencies(keys=[ord("r"), ord("q")])
        report = app.run(self.args(headless=False))
        self.assertEqual(analyzer.resets, 1)
        self.assertEqual(report["target_id_generations"], [(1, 1), (1, 2)])
        row = report["csv_rows"][0]
        self.assertTrue(row["Mixed_Identity"])
        self.assertEqual(row["Status"], "unknown")
        self.assertEqual(row["Discarded_Frames"], 1)

    def test_frame_limit_and_seek_use_original_source_timestamps(self):
        capture, analyzer, _, _ = self.dependencies(capture=Capture(frames=10))
        report = app.run(self.args("--start-seconds", "8", "--max-frames", "2"))
        self.assertEqual(analyzer.timestamps, [8000, 8500])
        self.assertEqual(capture.read_count, 2)
        self.assertEqual(report["end_reason"], "frame_limit")
        self.assertEqual(report["csv_rows"][0]["Time_Window"], "08s - 09s")

    def test_unknown_observation_is_not_logged_as_none(self):
        self.dependencies(analyzer=Analyzer(playing=None))
        report = app.run(self.args())
        self.assertEqual(report["metrics"]["unknown_frames"], 3)
        self.assertEqual(report["csv_rows"][0]["Status"], "unknown")

    def test_output_aliasing_an_input_is_rejected_before_opening_video(self):
        _, _, _, mocks = self.dependencies()
        args = self.args()
        args.output_csv = self.input
        with self.assertRaisesRegex(ValueError, "distinct"):
            app.run(args)
        mocks["VideoCapture"].assert_not_called()
        self.assertEqual(self.input.read_bytes(), b"")

    def test_duplicate_report_and_csv_are_rejected_before_opening_video(self):
        _, _, _, mocks = self.dependencies()
        args = self.args()
        args.report = args.output_csv
        with self.assertRaisesRegex(ValueError, "distinct"):
            app.run(args)
        mocks["VideoCapture"].assert_not_called()

    def test_model_error_reports_failure_and_releases_every_resource(self):
        capture, analyzer, writer, _ = self.dependencies(analyzer=Analyzer(fail_at=1))
        with self.assertRaisesRegex(RuntimeError, "mock model inference failed"):
            app.run(self.args("--export-video", str(self.root / "demo.mp4")))
        self.assertTrue(capture.released and analyzer.closed and writer.released)
        report = json.loads(self.report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "failed")
        self.assertIn("mock model inference failed", report["error"])

    def test_model_error_preserves_the_already_observed_partial_window(self):
        self.dependencies(analyzer=Analyzer(fail_at=1))
        with self.assertRaisesRegex(RuntimeError, "mock model inference failed"):
            app.run(self.args())
        report = json.loads(self.report_path.read_text(encoding="utf-8"))
        self.assertEqual(len(report["csv_rows"]), 1)
        self.assertEqual(report["csv_rows"][0]["Valid_Frames"], 1)
        self.assertTrue(report["csv_rows"][0]["Partial_Window"])

    def test_cleanup_error_cannot_leave_a_successful_execution_report(self):
        capture, analyzer, writer, _ = self.dependencies(analyzer=Analyzer(close_error="mock graph close failed"))
        with self.assertRaisesRegex(RuntimeError, "mock graph close failed"):
            app.run(self.args("--export-video", str(self.root / "demo.mp4")))
        self.assertTrue(capture.released and analyzer.closed and writer.released)
        report = json.loads(self.report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "failed")
        self.assertIn("mock graph close failed", json.dumps(report))

    def test_closed_native_window_is_a_normal_quit_signal(self):
        with patch.object(app.cv2, "getWindowProperty", side_effect=app.cv2.error("window closed")):
            self.assertFalse(app.window_visible())

    def test_invalid_cli_time_limit_and_overlap_fail_before_loading_models(self):
        for options in (("--start-seconds", "nan"), ("--max-frames", "0"),
                        ("--iou-threshold", "1.1")):
            with self.subTest(options=options), patch.object(app, "run") as runner:
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    app.main(options)
                self.assertEqual(error.exception.code, 2)
                runner.assert_not_called()


if __name__ == "__main__":
    unittest.main()
