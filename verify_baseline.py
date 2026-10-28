"""Run the original attention.main() on a bounded video sample.

This checks execution, not full-video completion or detection accuracy. The ROI
is the entire frame, supplied automatically instead of being selected by hand.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
from pathlib import Path
import sys
import time
import traceback
from contextlib import ExitStack
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT / "baseline_verification" / "source"
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache" / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--output-dir", default="baseline_verification")
    parser.add_argument("--gui-smoke", action="store_true")
    args = parser.parse_args()
    if args.frames < 1 or not math.isfinite(args.start_sec) or args.start_sec < 0:
        parser.error("--frames must be positive; --start-sec must be finite and nonnegative")

    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "verification_report.json"
    csv_path = output_dir / "action_log.csv"
    started = time.perf_counter()
    report = {
        "status": "failed",
        "scope": "Bounded execution smoke test; not full-video or accuracy validation.",
        "roi_policy": "Automatically use the full frame; no manual subject selection.",
        "arguments": vars(args),
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "source_sha256": {},
        "video": {},
        "metrics": {
            "frames_read": 0,
            "yolo_calls": 0,
            "frames_with_toy_boxes": 0,
            "toy_boxes_total": 0,
            "mediapipe_calls": 0,
            "frames_with_pose": 0,
            "frames_with_left_hand": 0,
            "frames_with_right_hand": 0,
            "frames_with_any_hand": 0,
            "iou_calls": 0,
            "positive_iou_calls": 0,
            "frames_with_positive_iou": 0,
            "automatic_roi_calls": 0,
        },
        "gui_smoke": "not_requested",
        "csv_path": str(csv_path),
        "csv_rows": [],
    }
    captures, holistics = [], []
    positive_frames = set()
    metrics = report["metrics"]

    try:
        for package in ("numpy", "opencv-python", "opencv-contrib-python",
                        "opencv-python-headless", "mediapipe", "ultralytics", "torch"):
            try:
                report["environment"][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                report["environment"][package] = None
        for filename in ("attention.py", "video-save.py"):
            report["source_sha256"][filename] = hashlib.sha256(
                (SOURCE_ROOT / filename).read_bytes()
            ).hexdigest()

        spec = importlib.util.spec_from_file_location("attention_baseline", SOURCE_ROOT / "attention.py")
        baseline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline)
        cv2 = baseline.cv2
        baseline.VIDEO_PATH = str(ROOT / "DD-22-PZ-ST0022-20210325.mp4")
        baseline.MODEL_PATH = str(ROOT / "best.pt")
        baseline.OUTPUT_CSV = str(csv_path)
        for path in (baseline.VIDEO_PATH, baseline.MODEL_PATH):
            if not Path(path).is_file():
                raise FileNotFoundError(path)
        real_capture = cv2.VideoCapture
        real_yolo = baseline.YOLO
        real_holistic = baseline.mp.solutions.holistic.Holistic
        real_iou = baseline.calculate_iou

        if args.gui_smoke:
            report["gui_smoke"] = "failed"
            preview = real_capture(baseline.VIDEO_PATH)
            try:
                ok, frame = preview.read()
                if not ok:
                    raise RuntimeError("Cannot read a frame for the GUI smoke test")
                cv2.namedWindow("Baseline GUI smoke", cv2.WINDOW_NORMAL)
                cv2.imshow("Baseline GUI smoke", frame)
                cv2.waitKey(200)
            finally:
                preview.release()
                cv2.destroyAllWindows()
            report["gui_smoke"] = "passed"

        class Capture:
            def __init__(self, *capture_args, **capture_kwargs):
                self.inner = real_capture(*capture_args, **capture_kwargs)
                captures.append(self.inner)
                if not self.inner.isOpened():
                    raise RuntimeError("OpenCV could not open the input video")
                fps = self.inner.get(cv2.CAP_PROP_FPS)
                count = self.inner.get(cv2.CAP_PROP_FRAME_COUNT)
                report["video"] = {
                    "path": baseline.VIDEO_PATH,
                    "fps": fps,
                    "width": int(self.inner.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(self.inner.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    "total_frames": count,
                    "duration_sec": count / fps if fps > 0 else None,
                    "requested_start_sec": args.start_sec,
                }
                if args.start_sec and not self.inner.set(cv2.CAP_PROP_POS_MSEC, args.start_sec * 1000):
                    raise RuntimeError("OpenCV could not seek to --start-sec")
                report["video"]["start_position_frames"] = self.inner.get(cv2.CAP_PROP_POS_FRAMES)

            def read(self, *read_args, **read_kwargs):
                if metrics["frames_read"] >= args.frames:
                    return False, None
                ok, frame = self.inner.read(*read_args, **read_kwargs)
                if ok:
                    metrics["frames_read"] += 1
                return ok, frame

            def __getattr__(self, name):
                return getattr(self.inner, name)

        class Model:
            def __init__(self, *model_args, **model_kwargs):
                self.inner = real_yolo(*model_args, **model_kwargs)
                report["model"] = {"path": baseline.MODEL_PATH, "classes": self.inner.names}

            def __call__(self, *model_args, **model_kwargs):
                results = self.inner(*model_args, **model_kwargs)
                boxes = sum(len(r.boxes) for r in results if r.boxes is not None)
                metrics["yolo_calls"] += 1
                metrics["frames_with_toy_boxes"] += int(boxes > 0)
                metrics["toy_boxes_total"] += boxes
                report["model"]["device"] = str(self.inner.device)
                return results

        class Holistic:
            def __init__(self, *holistic_args, **holistic_kwargs):
                self.inner = real_holistic(*holistic_args, **holistic_kwargs)
                holistics.append(self.inner)

            def process(self, frame):
                result = self.inner.process(frame)
                left = result.left_hand_landmarks is not None
                right = result.right_hand_landmarks is not None
                metrics["mediapipe_calls"] += 1
                metrics["frames_with_pose"] += int(result.pose_landmarks is not None)
                metrics["frames_with_left_hand"] += int(left)
                metrics["frames_with_right_hand"] += int(right)
                metrics["frames_with_any_hand"] += int(left or right)
                return result

        def iou(*boxes):
            value = real_iou(*boxes)
            metrics["iou_calls"] += 1
            if value > 0:
                metrics["positive_iou_calls"] += 1
                positive_frames.add(metrics["frames_read"])
            return value

        def full_frame_roi(_window, frame, **_kwargs):
            metrics["automatic_roi_calls"] += 1
            return (0, 0, frame.shape[1], frame.shape[0])

        with ExitStack() as stack:
            for name in ("namedWindow", "resizeWindow", "imshow", "destroyAllWindows"):
                stack.enter_context(patch.object(cv2, name, return_value=None))
            stack.enter_context(patch.object(cv2, "waitKey", return_value=-1))
            stack.enter_context(patch.object(cv2, "selectROI", side_effect=full_frame_roi))
            stack.enter_context(patch.object(cv2, "VideoCapture", Capture))
            stack.enter_context(patch.object(baseline, "YOLO", Model))
            stack.enter_context(patch.object(baseline.mp.solutions.holistic, "Holistic", Holistic))
            stack.enter_context(patch.object(baseline, "calculate_iou", side_effect=iou))
            baseline.main()

        if metrics["frames_read"] == 0:
            raise RuntimeError("No frames were processed; check the video and --start-sec")
        if metrics["yolo_calls"] != metrics["frames_read"] or metrics["mediapipe_calls"] != metrics["frames_read"]:
            raise RuntimeError("Not all sampled frames completed both inference stages")
        with csv_path.open(newline="", encoding="utf-8") as stream:
            report["csv_rows"] = list(csv.reader(stream))
        if not report["csv_rows"] or report["csv_rows"][0] != ["Time_Window", "Status"]:
            raise RuntimeError("The original program did not write the expected CSV header")
        report["status"] = "passed"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
    finally:
        for resource, method in [(item, "release") for item in captures] + [(item, "close") for item in holistics]:
            try:
                getattr(resource, method)()
            except Exception as error:
                report.setdefault("cleanup_errors", []).append(str(error))
                report["status"] = "failed"
        metrics["frames_with_positive_iou"] = len(positive_frames)
        report["elapsed_sec"] = round(time.perf_counter() - started, 3)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Baseline verification {report['status']}: {report_path}")
    if "error" in report:
        print(report["error"], file=sys.stderr)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
