"""Shoulder-selected OC-SORT target and target-only hand/toy analysis."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache" / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import cv2
import mediapipe as mp
from ultralytics import YOLO

from attention_engine import AttentionAnalyzer
from attention_log import ActionWindowLog
from person_detector import PersonDetector, DEFAULT_DETECTOR_MODEL
from tracking_pipeline import PipelineConfig, TrackingPipeline, DEFAULT_MODEL, ensure_model

VIDEO_PATH = ROOT / "DD-22-PZ-ST0022-20210325.mp4"
MODEL_PATH = ROOT / "best.pt"
OUTPUT_CSV = ROOT / "action_log.csv"
WINDOW = "Toy Tracking & Analysis - OC-SORT"


def window_visible():
    try:
        return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False


def project_path(value):
    path = Path(value).expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def draw_analysis(frame, result):
    """Draw only after all models have consumed the unannotated source frame."""
    canvas = frame.copy()
    for track in result.tracking.tracks:
        selected = track.track_id == result.tracking.target_id
        color = (0, 255, 255) if selected else ((190, 190, 190) if track.observed else (90, 90, 90))
        x1, y1, x2, y2 = map(int, track.bbox)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3 if selected else 1)
        label = f"ID {track.track_id}" + (" TARGET" if selected else "")
        if not track.observed:
            label += " predicted"
        cv2.putText(canvas, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, .65, color, 2)
    for box in result.toy_boxes:
        cv2.rectangle(canvas, tuple(box[:2]), tuple(box[2:]), (0, 255, 0), 2)
    if result.holistic is not None:
        left, top, width, height = result.crop_rect
        crop = canvas[top:top + height, left:left + width]
        drawing, styles = mp.solutions.drawing_utils, mp.solutions.drawing_styles
        holistic = mp.solutions.holistic
        if result.holistic.face_landmarks:
            drawing.draw_landmarks(crop, result.holistic.face_landmarks,
                                   holistic.FACEMESH_TESSELATION, landmark_drawing_spec=None,
                                   connection_drawing_spec=styles.get_default_face_mesh_tesselation_style())
        drawing.draw_landmarks(crop, result.holistic.pose_landmarks, holistic.POSE_CONNECTIONS,
                               landmark_drawing_spec=styles.get_default_pose_landmarks_style())
        for hand in (result.holistic.left_hand_landmarks, result.holistic.right_hand_landmarks):
            if hand:
                drawing.draw_landmarks(crop, hand, holistic.HAND_CONNECTIONS,
                                       landmark_drawing_spec=styles.get_default_hand_landmarks_style())
    for box in result.hand_boxes:
        cv2.rectangle(canvas, tuple(box[:2]), tuple(box[2:]), (255, 0, 0), 2)
    colors = {"playing": (0, 255, 255), "none": (0, 0, 255), "unknown": (190, 190, 190)}
    cv2.putText(canvas, result.status, (max(10, canvas.shape[1] - 245), 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, colors[result.status], 3)
    lines = [f"{result.tracking.state} | ID {result.tracking.target_id} | generation {result.tracking.selection_generation}",
             result.detail, "Q/ESC: quit   SPACE: pause   R: reselect by shoulders"]
    if result.tracking.target_id is None:
        for track_id, values in result.tracking.statistics.items():
            width = values.get("median_width")
            label = "n/a" if width is None else f"{width:.1f}px"
            lines.append(f"ID {track_id}: shoulders {label}, samples {values.get('sample_count', 0)}, valid {values.get('valid_seconds', 0):.1f}s")
    for index, text in enumerate(lines):
        cv2.putText(canvas, text, (15, 30 + index * 28), cv2.FONT_HERSHEY_SIMPLEX,
                    .65, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def build_parser(export_default=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=project_path, default=VIDEO_PATH)
    parser.add_argument("--model", type=project_path, default=MODEL_PATH, help="Trained toy YOLO weights")
    parser.add_argument("--output-csv", type=project_path, default=OUTPUT_CSV)
    parser.add_argument("--export-video", type=project_path, default=export_default)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--report", type=project_path, help="Optional execution statistics JSON")
    parser.add_argument("--iou-threshold", type=float, default=0.0)
    parser.add_argument("--init-seconds", type=float, default=2.0)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--min-gap", type=float, default=0.10)
    parser.add_argument("--landmark-confidence", type=float, default=0.70)
    parser.add_argument("--auto-reset-seconds", type=float, default=0.0)
    parser.add_argument("--pose-recovery-seconds", type=float, default=2.0)
    parser.add_argument("--pose-reselect-seconds", type=float, default=10.0)
    parser.add_argument("--no-reselect-on-loss", action="store_true")
    return parser


def run(args):
    for path in (args.input, args.model):
        if not path.is_file():
            raise FileNotFoundError(path)
    outputs = [args.output_csv] + [path for path in (args.export_video, args.report) if path is not None]
    inputs = {args.input.resolve(), args.model.resolve(), DEFAULT_MODEL.resolve(), DEFAULT_DETECTOR_MODEL.resolve()}
    if len(set(path.resolve() for path in outputs)) != len(outputs) or any(path.resolve() in inputs for path in outputs):
        raise ValueError("Input, model and output paths must be distinct")
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(args.input))
    analyzer = writer = pipeline = None
    started = time.perf_counter()
    metrics = Counter()
    states, details, identities = Counter(), Counter(), set()
    report = {"status": "failed", "input": str(args.input), "model": str(args.model)}
    window_log = None
    last = None
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {args.input}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            fps = 30.0
        width, height = (int(cap.get(key)) for key in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
        if args.start_seconds and not cap.set(cv2.CAP_PROP_POS_MSEC, args.start_seconds * 1000):
            raise RuntimeError("Cannot seek to the requested source-video time")
        start_frame = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))
        config = PipelineConfig(init_seconds=args.init_seconds, min_samples=args.min_samples,
                                min_gap=args.min_gap, landmark_confidence=args.landmark_confidence,
                                auto_reset_seconds=args.auto_reset_seconds,
                                pose_recovery_seconds=args.pose_recovery_seconds,
                                pose_reselect_seconds=args.pose_reselect_seconds,
                                retain_target_on_loss=args.no_reselect_on_loss,
                                require_posture_landmarks=False)
        print("Loading YOLOX person detector, OC-SORT, shoulder selector and toy YOLO...", flush=True)
        detector = PersonDetector()
        toy_model = YOLO(str(args.model))
        pipeline = TrackingPipeline(detector, ensure_model(DEFAULT_MODEL), fps, config)
        analyzer = AttentionAnalyzer(toy_model, pipeline, fps, iou_threshold=args.iou_threshold)
        report.update({"video": {"fps": fps, "width": width, "height": height, "start_frame": start_frame,
                                 "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))},
                       "tracking_config": asdict(config), "iou_threshold": args.iou_threshold,
                       "person_detector": detector.provenance})
        if args.export_video:
            writer = cv2.VideoWriter(str(args.export_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"Cannot create output video: {args.export_video}")
        if not args.headless:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW, 1280, 720)
        print("Selecting by shoulder width automatically. No manual ROI is needed.", flush=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
            window_log = ActionWindowLog(stream, fps, start_seconds=start_frame / fps)
            try:
                while True:
                    if args.max_frames is not None and metrics["frames"] >= args.max_frames:
                        report["end_reason"] = "frame_limit"
                        break
                    ok, frame = cap.read()
                    if not ok:
                        report["end_reason"] = "eof"
                        break
                    timestamp = round((start_frame + metrics["frames"]) * 1000 / fps)
                    last = analyzer.process(frame, timestamp)
                    window_log.add(last)
                    metrics["frames"] += 1
                    metrics["frames_with_hands"] += bool(last.hand_boxes)
                    metrics["frames_with_toys"] += bool(last.toy_boxes)
                    metrics["playing_frames"] += last.playing is True
                    metrics["unknown_frames"] += last.playing is None
                    states[last.tracking.state] += 1
                    details[last.detail] += 1
                    if last.tracking.target_id is not None:
                        identities.add((last.tracking.target_id, last.tracking.selection_generation))
                    if writer is not None or not args.headless:
                        canvas = draw_analysis(frame, last)
                        if writer is not None:
                            writer.write(canvas)
                        if not args.headless:
                            cv2.imshow(WINDOW, canvas)
                            key = cv2.waitKey(1) & 0xFF
                            if key == ord(" "):
                                while window_visible():
                                    key = cv2.waitKey(50) & 0xFF
                                    if key != 255:
                                        break
                                if key == ord(" "):
                                    key = -1
                            if key in (ord("q"), ord("Q"), 27) or not window_visible():
                                report["end_reason"] = "user_quit"
                                break
                            if key in (ord("r"), ord("R")):
                                analyzer.reset_selection()
                    if metrics["frames"] % 150 == 0:
                        print(f"Frames {metrics['frames']} | target {last.tracking.target_id} | {last.tracking.state} | {last.status}", flush=True)
            finally:
                window_log.finish()
        if not metrics["frames"]:
            raise RuntimeError("No source frames were processed")
        report["status"] = "passed"
        report["toy_device"] = str(toy_model.device)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        active_error = sys.exc_info()[0] is not None
        cleanup_errors = []
        cleanup = [("capture", cap.release)]
        if analyzer is not None:
            cleanup.insert(0, ("analyzer", analyzer.close))
        elif pipeline is not None:
            cleanup.insert(0, ("pipeline", pipeline.close))
        if writer is not None:
            cleanup.append(("video_writer", writer.release))
        if not args.headless:
            cleanup.append(("windows", cv2.destroyAllWindows))
        for name, close in cleanup:
            try:
                close()
            except Exception as error:
                cleanup_errors.append(f"{name}: {type(error).__name__}: {error}")
        if cleanup_errors:
            report["status"] = "failed"
            report["cleanup_errors"] = cleanup_errors
        report.update({"metrics": dict(metrics), "tracking_states": dict(states), "details": dict(details),
                       "target_id_generations": sorted(identities),
                       "elapsed_seconds": round(time.perf_counter() - started, 3),
                       "csv_rows": window_log.rows if window_log is not None else []})
        if last is not None:
            report["last_tracking"] = {"target_id": last.tracking.target_id,
                                        "selection_generation": last.tracking.selection_generation,
                                        "state": last.tracking.state,
                                        "statistics": last.tracking.statistics}
        if args.report:
            args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if cleanup_errors and not active_error:
            raise RuntimeError("; ".join(cleanup_errors))
    print(f"Finished: {metrics['frames']} frames; CSV: {args.output_csv}", flush=True)
    return report


def main(argv=None, *, export_default=None):
    parser = build_parser(export_default)
    args = parser.parse_args(argv)
    if not math.isfinite(args.start_seconds) or args.start_seconds < 0:
        parser.error("--start-seconds must be finite and nonnegative")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    if not math.isfinite(args.iou_threshold) or not 0 <= args.iou_threshold <= 1:
        parser.error("--iou-threshold must lie between zero and one")
    try:
        run(args)
    except (OSError, RuntimeError, ValueError, cv2.error) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
