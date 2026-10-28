"""Identity, geometry and temporal checks without loading any real models."""

from types import SimpleNamespace as NS
import unittest

import numpy as np

from attention_engine import AttentionAnalyzer, calculate_iou


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def __getitem__(self, key):
        return Tensor(self.values[key])

    def cpu(self):
        return self

    def numpy(self):
        return self.values.copy()

    def tolist(self):
        return self.values.tolist()


class ToyModel:
    def __init__(self, per_frame):
        self.outputs = iter(per_frame)
        self.inputs = []

    def __call__(self, frame, verbose=False):
        self.inputs.append(frame.copy())
        return [NS(boxes=[NS(xyxy=Tensor([box])) for box in next(self.outputs)])]


class Tracking:
    def __init__(self, per_frame):
        self.outputs = iter(per_frame)
        self.inputs = []
        self.indices = []
        self.closed = False
        self.config = NS(crop_padding=0.08, landmark_confidence=0.7)

    def process(self, frame, timestamp_ms, frame_index):
        self.inputs.append(frame.copy())
        self.indices.append(frame_index)
        return next(self.outputs)

    def close(self):
        self.closed = True


def analysis(target_id=1, generation=1, state="locked", observed=True):
    target = NS(track_id=target_id, bbox=np.array([100, 100, 300, 300]),
                observed=observed)
    other = NS(track_id=99, bbox=np.array([0, 0, 60, 80]), observed=True)
    points = np.zeros((33, 2), dtype=float)
    points[11] = (142, 142)
    points[12] = (258, 142)
    pose = NS(points=points, visibility=np.ones(33), presence=np.ones(33))
    return NS(tracks=[other, target], target_id=target_id,
              pose=pose if state == "locked" else None, state=state,
              detail=state, selection_generation=generation)


def holistic_result(*, hands=True, confidence=1.0, shoulder_offset=0.0):
    pose = [NS(x=0.5, y=0.5, visibility=1.0) for _ in range(33)]
    pose[11] = NS(x=0.25 + shoulder_offset, y=0.25, visibility=confidence)
    pose[12] = NS(x=0.75 + shoulder_offset, y=0.25, visibility=confidence)
    hand = NS(landmark=[NS(x=float(x), y=float(y)) for x, y in
                        zip(np.linspace(0.4, 0.6, 21), np.linspace(0.6, 0.8, 21))])
    return NS(pose_landmarks=NS(landmark=pose), face_landmarks=None,
              left_hand_landmarks=hand if hands else None,
              right_hand_landmarks=None)


class HolisticFactory:
    def __init__(self, results=None):
        self.results = iter(results) if results is not None else None
        self.instances = []
        self.inputs = []

    def __call__(self, **_options):
        owner = self

        class Holistic:
            closed = False

            def process(self, frame):
                owner.inputs.append(frame.copy())
                return next(owner.results) if owner.results is not None else holistic_result()

            def close(self):
                self.closed = True

        instance = Holistic()
        self.instances.append(instance)
        return instance


class AttentionEngineTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((400, 500, 3), [50, 100, 150], dtype=np.uint8)
        self.toy = [190, 230, 210, 250]

    def analyzer(self, analyses, toys, results=None, **options):
        factory = HolisticFactory(results)
        tracking = Tracking(analyses)
        model = ToyModel(toys)
        engine = AttentionAnalyzer(model, tracking, fps=30,
                                   holistic_factory=factory, **options)
        self.addCleanup(engine.close)
        return engine, factory, tracking, model

    def test_target_crop_hands_are_mapped_back_to_original_pixels(self):
        engine, factory, _, _ = self.analyzer([analysis()], [[self.toy]])
        result = engine.process(self.frame, 0)
        self.assertEqual(result.status, "playing")
        self.assertEqual(tuple(result.crop_rect), (84, 84, 232, 232))
        np.testing.assert_array_equal(result.target_bbox, [100, 100, 300, 300])
        self.assertEqual(len(result.hand_boxes), 1)
        np.testing.assert_allclose(result.hand_boxes[0], [174, 220, 225, 271], atol=1)
        self.assertEqual(factory.inputs[0].shape, (232, 232, 3))

    def test_no_target_or_unobserved_target_never_uses_another_person(self):
        for state, observed in (("collecting", True), ("lost", False), ("locked", False)):
            with self.subTest(state=state, observed=observed):
                item = analysis(state=state, observed=observed)
                if state == "collecting":
                    item.target_id = None
                engine, factory, _, model = self.analyzer([item], [[self.toy]])
                result = engine.process(self.frame, 0)
                self.assertEqual(result.status, "unknown")
                self.assertIsNone(result.playing)
                self.assertEqual(result.hand_boxes, [])
                self.assertEqual(factory.inputs, [])
                self.assertEqual(len(model.inputs), 1)

    def test_reselection_discards_previous_target_toys_and_temporal_graph(self):
        for changed in (analysis(target_id=2), analysis(generation=2)):
            with self.subTest(target=changed.target_id, generation=changed.selection_generation):
                engine, factory, _, _ = self.analyzer(
                    [analysis(), changed], [[self.toy], []])
                self.assertEqual(engine.process(self.frame, 0).status, "playing")
                self.assertEqual(engine.process(self.frame, 33).status, "none")
                self.assertTrue(factory.instances[0].closed)
                self.assertEqual(len(factory.instances), 2)

    def test_lost_frame_discards_history_before_same_id_returns(self):
        engine, factory, _, _ = self.analyzer(
            [analysis(), analysis(state="lost", observed=False), analysis()],
            [[self.toy], [self.toy], []])
        self.assertEqual(engine.process(self.frame, 0).status, "playing")
        self.assertEqual(engine.process(self.frame, 33).status, "unknown")
        self.assertTrue(factory.instances[0].closed)
        self.assertEqual(engine.process(self.frame, 66).status, "none")
        self.assertEqual(len(factory.instances), 2)

    def test_collecting_toys_are_not_reused_after_first_lock(self):
        engine, _, _, _ = self.analyzer(
            [analysis(state="collecting"), analysis()], [[self.toy], []])
        self.assertEqual(engine.process(self.frame, 0).status, "unknown")
        self.assertEqual(engine.process(self.frame, 33).status, "none")

    def test_short_history_bridges_temporary_missing_toy_detections(self):
        engine, _, _, _ = self.analyzer([analysis(), analysis()], [[self.toy], []])
        self.assertTrue(engine.process(self.frame, 0).playing)
        self.assertTrue(engine.process(self.frame, 33).playing)

    def test_history_expires_after_configured_number_of_source_frames(self):
        engine, _, _, _ = self.analyzer(
            [analysis()] * 16, [[self.toy]] + [[]] * 15)
        results = [engine.process(self.frame, round(i * 1000 / 30)) for i in range(16)]
        self.assertTrue(results[14].playing)
        self.assertFalse(results[15].playing)

    def test_holistic_must_agree_with_tasks_shoulders_before_hands_are_accepted(self):
        for bad in (holistic_result(confidence=0.1),
                    holistic_result(shoulder_offset=0.6)):
            with self.subTest(bad=bad):
                engine, factory, _, _ = self.analyzer([analysis()], [[self.toy]], [bad])
                result = engine.process(self.frame, 0)
                self.assertEqual(result.status, "unknown")
                self.assertEqual(result.hand_boxes, [])
                self.assertIsNone(result.holistic)
                self.assertTrue(factory.instances[0].closed)

    def test_rejected_holistic_does_not_leave_old_toy_history(self):
        engine, _, _, _ = self.analyzer(
            [analysis()] * 3, [[self.toy], [], []],
            [holistic_result(), holistic_result(confidence=0), holistic_result()])
        self.assertEqual(engine.process(self.frame, 0).status, "playing")
        self.assertEqual(engine.process(self.frame, 33).status, "unknown")
        self.assertEqual(engine.process(self.frame, 66).status, "none")

    def test_absent_hands_with_valid_target_are_none(self):
        engine, _, _, _ = self.analyzer([analysis()], [[self.toy]],
                                         [holistic_result(hands=False)])
        self.assertEqual(engine.process(self.frame, 0).status, "none")

    def test_custom_iou_threshold_is_strict(self):
        engine, _, _, _ = self.analyzer([analysis()], [[self.toy]], iou_threshold=1.0)
        self.assertEqual(engine.process(self.frame, 0).status, "none")
        self.assertEqual(calculate_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertEqual(calculate_iou([0, 0, 10, 10], [10, 0, 20, 10]), 0.0)

    def test_models_receive_unannotated_input_and_holistic_receives_rgb(self):
        original = self.frame.copy()
        engine, factory, tracking, model = self.analyzer([analysis()], [[self.toy]])
        engine.process(self.frame, 0)
        np.testing.assert_array_equal(self.frame, original)
        np.testing.assert_array_equal(tracking.inputs[0], original)
        np.testing.assert_array_equal(model.inputs[0], original)
        np.testing.assert_array_equal(factory.inputs[0][0, 0], [150, 100, 50])
        self.assertEqual(len(model.inputs), 1)

    def test_close_releases_both_temporal_graphs(self):
        engine, factory, tracking, _ = self.analyzer([analysis()], [[]])
        engine.process(self.frame, 0)
        engine.close()
        self.assertTrue(factory.instances[0].closed)
        self.assertTrue(tracking.closed)


if __name__ == "__main__":
    unittest.main()
