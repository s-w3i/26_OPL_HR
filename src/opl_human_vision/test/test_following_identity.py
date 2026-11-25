import importlib.util
from pathlib import Path
import time
import unittest

import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from opl_interfaces.msg import TrackedHuman, TrackedHumanArray, FollowingTarget
from opl_interfaces.srv import SelectTarget

spec = importlib.util.spec_from_file_location('following', Path(__file__).parents[1] / 'scripts/following_target.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FollowingIdentityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = module.FollowingTargetNode()
        self.probe = Node('following_test_probe')
        self.transforms = []
        self.sub = self.probe.create_subscription(TFMessage, '/tf',
            lambda msg: self.transforms.extend(msg.transforms), 100)
        self.a = self.person(8, 2, 'furia')
        self.b = self.person(9, 3, 'other')

    def tearDown(self):
        self.node.destroy_node()
        self.probe.destroy_node()

    def person(self, raw, identity, name):
        h = TrackedHuman(raw_track_id=raw, track_id=identity, name=name,
                         observed=True, position_valid=True, identity_state=TrackedHuman.FACE_CONFIRMED)
        h.position_3d.z = 2.0
        return h

    def message(self, humans):
        msg = TrackedHumanArray(humans=humans)
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_color_optical_frame'
        return msg

    def observe(self, humans):
        msg = self.message(humans)
        self.node.tracks(msg)
        self.node.recognized(msg)
        return msg

    def select(self, identity):
        return self.node.select(SelectTarget.Request(identity_id=identity), SelectTarget.Response())

    def collect(self):
        end = time.monotonic() + 0.06
        while time.monotonic() < end:
            rclpy.spin_once(self.probe, timeout_sec=0.005)

    def test_prediction_flags_only_authorize_ordinary_missed_detection(self):
        output=[]
        self.node.pub=type('Capture',(),{'publish':lambda _,m:output.append(m)})()
        self.observe([self.a]);self.select(2)
        self.assertTrue(output[-1].observed)
        self.assertFalse(output[-1].prediction_allowed)
        stamp=output[-1].header.stamp
        self.a.observed=False
        self.observe([self.a])
        self.assertFalse(output[-1].observed)
        self.assertTrue(output[-1].prediction_allowed)
        self.assertEqual(output[-1].header.stamp,stamp)
        self.node.last_receive-=.16;self.node.watchdog()
        self.assertFalse(output[-1].prediction_allowed)

    def test_navigation_absence_outlives_prediction_and_revokes_on_stream_failure(self):
        from unittest.mock import patch

        status = []
        targets = []
        self.node.prediction_status = type('Capture', (), {'publish': lambda _, msg: status.append(msg)})()
        self.node.pub = type('Capture', (), {'publish': lambda _, msg: targets.append(msg)})()

        def values():
            return {value.key: value.value for value in status[-1].status[0].values}

        self.observe([self.a])
        self.assertTrue(self.select(2).success)
        selection = values().get('selection_generation')
        self.assertTrue(selection)
        self.assertEqual(values().get('navigation_visibility'), 'observed')
        self.observe([])
        self.assertEqual(values().get('navigation_visibility'), 'absent')
        original = self.node.get_clock().now().nanoseconds
        for frame in range(1, 14):
            now = rclpy.time.Time(nanoseconds=original + frame * 250_000_000)
            with patch.object(self.node.get_clock(), 'now', return_value=now):
                self.observe([])
        self.assertFalse(targets[-1].prediction_allowed)
        self.assertEqual(values().get('navigation_visibility'), 'absent')
        self.assertEqual(values().get('selection_generation'), selection)
        with patch.object(self.node.get_clock(), 'now', return_value=now):
            self.node.last_receive -= .2
            self.node.watchdog()
            self.assertEqual(values().get('navigation_visibility'), 'invalid')
        with patch.object(self.node.get_clock(), 'now', return_value=now + rclpy.time.Duration(seconds=.1)):
            self.observe([])
            self.assertEqual(values().get('navigation_visibility'), 'invalid')
        self.select(0)
        self.assertNotEqual(values().get('selection_generation'), selection)

    def test_navigation_keeps_same_identity_on_depth_loss_and_revokes_on_conflict(self):
        from unittest.mock import patch

        status = []
        self.node.prediction_status = type('Capture', (), {'publish': lambda _, msg: status.append(msg)})()

        def visibility():
            return {value.key: value.value for value in status[-1].status[0].values}['navigation_visibility']

        self.observe([self.a])
        self.assertTrue(self.select(2).success)
        self.assertEqual(visibility(), 'observed')
        self.a.position_valid = False
        original = self.node.get_clock().now().nanoseconds
        for frame in range(1, 14):
            now = rclpy.time.Time(nanoseconds=original + frame * 250_000_000)
            with patch.object(self.node.get_clock(), 'now', return_value=now):
                self.observe([self.a])
                self.assertEqual(visibility(), 'position_unavailable')
        self.a.track_id = 3
        with patch.object(self.node.get_clock(), 'now', return_value=now + rclpy.time.Duration(seconds=.1)):
            self.observe([self.a])
            self.assertEqual(visibility(), 'invalid')

    def test_ordinary_loss_can_predict_past_750ms_but_stops_at_two_point_five_seconds(self):
        from unittest.mock import patch
        output=[]
        self.node.pub=type('Capture',(),{'publish':lambda _,m:output.append(m)})()
        self.observe([self.a]);self.select(2)
        original=self.node.get_clock().now().nanoseconds
        self.a.observed=False
        with patch.object(self.node.get_clock(),'now',return_value=rclpy.time.Time(nanoseconds=original+2_300_000_000)):
            self.observe([self.a])
            self.assertTrue(output[-1].prediction_allowed)
            self.assertFalse(output[-1].observed)
        with patch.object(self.node.get_clock(),'now',return_value=rclpy.time.Time(nanoseconds=original+2_510_000_000)):
            self.observe([self.a])
            self.assertFalse(output[-1].prediction_allowed)

    def test_unconfirmed_same_track_uses_only_last_measured_position_until_confirmed(self):
        output=[]
        self.node.pub=type('Capture',(),{'publish':lambda _,m:output.append(m)})()
        self.observe([self.a]);self.select(2)
        stamp=self.node.last_good[0].stamp
        original_z=self.node.last_good[1].position_3d.z
        self.a.identity_state=self.a.UNCONFIRMED;self.a.track_id=1000000001;self.a.name='unknown'
        self.node.recognized(self.message([self.a]))
        self.assertTrue(output[-1].prediction_allowed)
        self.assertFalse(output[-1].observed)
        self.assertFalse(self.select(2).success)
        self.a.position_3d.z=3.;self.observe([self.a])
        self.assertEqual(self.node.last_good[0].stamp,stamp)
        self.assertEqual(self.node.last_good[1].position_3d.z,original_z)
        self.a.identity_state=self.a.FACE_CONFIRMED;self.a.track_id=2;self.a.name='furia'
        self.observe([self.a])
        self.assertTrue(output[-1].observed)
        self.assertFalse(output[-1].prediction_allowed)
        self.assertEqual(self.node.last_good[1].position_3d.z,3.)

    def test_conflicting_identity_cannot_resume_an_unconfirmed_gap(self):
        output=[]
        self.node.pub=type('Capture',(),{'publish':lambda _,m:output.append(m)})()
        self.observe([self.a]);self.select(2)
        self.a.identity_state=self.a.UNCONFIRMED;self.a.track_id=1000000001
        self.node.recognized(self.message([self.a]))
        self.assertTrue(output[-1].prediction_allowed)
        self.a.identity_state=self.a.FACE_CONFIRMED;self.a.track_id=3
        self.observe([self.a])
        self.assertFalse(output[-1].prediction_allowed)
        self.assertFalse(output[-1].observed)
        self.assertEqual(self.node.identity,2)
        self.assertGreaterEqual(self.node.reacquire_after,0)

    def test_unconfirmed_gap_still_rejects_depth_failure_and_ambiguity(self):
        output=[]
        self.node.pub=type('Capture',(),{'publish':lambda _,m:output.append(m)})()
        self.observe([self.a]);self.select(2)
        self.a.identity_state=self.a.UNCONFIRMED;self.a.track_id=1000000001
        self.node.recognized(self.message([self.a]))
        self.a.position_valid=False;self.observe([self.a])
        self.assertFalse(output[-1].prediction_allowed)
        self.a.position_valid=True;self.a.association_ambiguous=True;self.observe([self.a])
        self.assertFalse(output[-1].prediction_allowed)

    def test_unconfirmed_gap_expires_without_refreshing_the_capture_stamp(self):
        from unittest.mock import patch
        output=[]
        self.node.pub=type('Capture',(),{'publish':lambda _,m:output.append(m)})()
        self.observe([self.a]);self.select(2)
        original=self.node.get_clock().now().nanoseconds
        self.a.identity_state=self.a.UNCONFIRMED;self.a.track_id=1000000001
        self.node.recognized(self.message([self.a]))
        with patch.object(self.node.get_clock(),'now',return_value=rclpy.time.Time(nanoseconds=original+2_400_000_000)):
            self.observe([self.a]);self.assertTrue(output[-1].prediction_allowed)
        with patch.object(self.node.get_clock(),'now',return_value=rclpy.time.Time(nanoseconds=original+2_510_000_000)):
            self.observe([self.a]);self.assertFalse(output[-1].prediction_allowed)

    def test_depth_failure_conflict_and_cancel_cannot_authorize_prediction(self):
        output=[]
        self.node.pub=type('Capture',(),{'publish':lambda _,m:output.append(m)})()
        self.observe([self.a]);self.select(2)
        self.a.position_valid=False;self.observe([self.a])
        self.assertFalse(output[-1].observed)
        self.assertFalse(output[-1].prediction_allowed)
        self.a.position_valid=True;self.a.association_ambiguous=True
        self.observe([self.a]);self.a.observed=False;self.observe([self.a])
        self.assertFalse(output[-1].prediction_allowed)
        self.select(0)
        self.assertFalse(output[-1].prediction_allowed)

    def test_tf_disabled_until_selection_and_cancel(self):
        self.observe([self.a, self.b])
        self.collect()
        self.assertEqual(self.transforms, [])
        self.assertTrue(self.select(2).success)
        self.collect()
        self.assertTrue(self.transforms)
        self.assertEqual({tf.child_frame_id for tf in self.transforms}, {'opl_hr_person_2'})
        self.assertEqual(self.transforms[-1].header.frame_id, 'camera_color_optical_frame')
        self.assertEqual(self.transforms[-1].transform.translation.z, 2.0)
        self.assertEqual(self.transforms[-1].transform.rotation.w, 1.0)
        self.assertTrue(self.select(0).success)
        self.transforms.clear()
        self.observe([self.a, self.b])
        self.collect()
        self.assertEqual(self.transforms, [])
        self.assertEqual(self.node.state, FollowingTarget.UNSELECTED)

    def test_reject_invalid_selection_without_replacing_target(self):
        self.observe([self.a, self.b])
        self.assertTrue(self.select(2).success)
        self.b.position_valid = False
        self.observe([self.a, self.b])
        self.assertFalse(self.select(3).success)
        self.assertFalse(self.select(99).success)
        self.assertFalse(self.select(-1).success)
        self.assertEqual(self.node.identity, 2)
        self.b.position_valid = True
        self.b.track_id = 1000000000
        self.b.identity_state = self.b.UNCONFIRMED
        self.observe([self.a, self.b])
        self.assertFalse(self.select(3).success)

    def test_covered_face_and_depth_recovery(self):
        self.observe([self.a])
        self.assertTrue(self.select(2).success)
        self.a.identity_state = self.a.TRACKED
        self.a.match_score = 0.0
        msg = self.observe([self.a])
        self.collect()
        self.assertEqual(self.node.state, FollowingTarget.TRACKING)
        self.assertEqual(self.transforms[-1].header.stamp, msg.header.stamp)
        self.a.position_valid = False
        original_stamp = self.node.last_good[0].stamp
        self.observe([self.a])
        self.assertEqual(self.node.state, FollowingTarget.TRACKING)
        self.assertEqual(self.node.last_good[0].stamp, original_stamp)
        self.node.last_good[0].stamp.sec -= 1
        self.observe([self.a])
        self.assertEqual(self.node.state, FollowingTarget.LOST)
        self.a.position_valid = True
        self.observe([self.a])
        self.assertEqual(self.node.state, FollowingTarget.TRACKING)

    def test_brief_missed_detection_preserves_identity_without_restamping(self):
        self.observe([self.a])
        self.select(2)
        stamp = self.node.last_good[0].stamp
        self.a.observed = False
        self.observe([self.a])
        self.assertEqual(self.node.state, FollowingTarget.TRACKING)
        self.assertEqual(self.node.last_good[0].stamp, stamp)
        self.assertEqual(self.node.reacquire_after, -1)
        self.a.observed = True
        self.a.identity_state = self.a.TRACKED
        self.observe([self.a])
        self.assertEqual(self.node.state, FollowingTarget.TRACKING)

    def test_duplicate_identity_and_clock_reset_invalidate_cached_position(self):
        from unittest.mock import patch
        self.observe([self.a])
        self.select(2)
        duplicate = self.person(11, 2, 'furia')
        self.observe([self.a, duplicate])
        self.assertEqual(self.node.state, FollowingTarget.LOST)
        self.assertIsNone(self.node.last_good)
        self.observe([self.a])
        self.select(2)
        previous = self.node.get_clock().now().nanoseconds
        with patch.object(self.node.get_clock(), 'now', return_value=rclpy.time.Time(nanoseconds=previous-1000000000)):
            self.node.watchdog()
        self.assertEqual(self.node.state, FollowingTarget.LOST)
        self.assertIsNone(self.node.last_good)

    def test_loss_requires_same_identity_new_face_on_new_raw_track(self):
        old = self.observe([self.a, self.b])
        self.select(2)
        self.node.last_good[0].stamp.sec -= 3
        self.observe([self.b])
        self.assertEqual(self.node.identity, 2)
        self.assertEqual(self.node.state, FollowingTarget.LOST)
        self.a.raw_track_id = 10
        self.a.identity_state = self.a.TRACKED
        self.observe([self.a, self.b])
        self.assertEqual(self.node.state, FollowingTarget.LOST)
        self.assertFalse(self.select(2).success)  # cannot bypass face reacquisition
        self.node.recognized(old)
        self.assertEqual(self.node.state, FollowingTarget.LOST)
        self.a.identity_state = self.a.FACE_CONFIRMED
        self.observe([self.a, self.b])
        self.assertEqual(self.node.selected, 10)
        self.assertEqual(self.node.state, FollowingTarget.TRACKING)
        self.assertEqual(self.node.identity, 2)

    def test_ambiguity_latches_and_different_face_never_replaces_target(self):
        old = self.observe([self.a, self.b])
        self.select(2)
        self.a.association_ambiguous = True
        self.observe([self.a, self.b])
        self.a.association_ambiguous = False
        self.a.identity_state = self.a.TRACKED
        self.observe([self.a, self.b])
        self.assertEqual(self.node.state, FollowingTarget.LOST)
        self.node.recognized(old)
        self.assertEqual(self.node.state, FollowingTarget.LOST)
        self.a.identity_state = self.a.FACE_CONFIRMED
        self.a.track_id = 3
        self.observe([self.a])
        self.assertEqual(self.node.identity, 2)
        self.assertEqual(self.node.state, FollowingTarget.LOST)
        self.a.track_id = 2
        self.observe([self.a])
        self.assertEqual(self.node.state, FollowingTarget.TRACKING)

    def test_explicit_switch_and_timeout(self):
        self.observe([self.a, self.b])
        self.select(2)
        self.assertTrue(self.select(3).success)
        self.collect()
        self.assertEqual(self.transforms[-1].child_frame_id, 'opl_hr_person_3')
        self.node.last_receive -= 1.0
        self.node.watchdog()
        self.assertEqual(self.node.state, FollowingTarget.LOST)

        self.assertEqual(self.node.identity, 3)
        self.b.identity_state = self.b.TRACKED
        self.observe([self.a, self.b])
        self.assertEqual(self.node.state, FollowingTarget.LOST)

    def test_brief_stale_pose_pauses_then_recovers_same_track(self):
        self.observe([self.a])
        self.assertTrue(self.select(2).success)
        self.a.identity_state = self.a.TRACKED
        self.node.last_receive -= 0.2
        self.node.last_header.stamp = rclpy.time.Time(
            nanoseconds=self.node.get_clock().now().nanoseconds - 200000000).to_msg()
        self.node.watchdog()
        self.assertEqual(self.node.state, FollowingTarget.LOST)
        self.assertEqual(self.node.reacquire_after, -1)
        self.observe([self.a])
        self.assertEqual(self.node.state, FollowingTarget.TRACKING)

    def test_new_fresh_observation_does_not_inherit_old_pose_timeout(self):
        self.observe([self.a])
        self.assertTrue(self.select(2).success)
        self.a.identity_state = self.a.TRACKED
        self.node.last_receive -= 0.2
        self.node.last_header.stamp = rclpy.time.Time(
            nanoseconds=self.node.get_clock().now().nanoseconds - 200000000).to_msg()
        self.observe([self.a])
        self.assertEqual(self.node.state, FollowingTarget.TRACKING)

    def test_long_gap_still_requires_face_reacquisition(self):
        self.observe([self.a])
        self.assertTrue(self.select(2).success)
        self.node.last_receive -= 0.2
        self.node.watchdog()
        self.assertEqual(self.node.state, FollowingTarget.LOST)
        self.assertEqual(self.node.reacquire_after, -1)
        self.node.last_receive -= 0.8
        self.node.watchdog()
        self.assertGreaterEqual(self.node.reacquire_after, 0)
        self.a.identity_state = self.a.TRACKED
        self.observe([self.a])
        self.assertEqual(self.node.state, FollowingTarget.LOST)

    def test_no_tf_updates_for_invalid_pose_prediction_or_stale_input(self):
        self.observe([self.a])
        self.select(2)
        self.collect()
        count = len(self.transforms)
        self.assertGreater(count, 0)
        for field, value in [('position_valid', False), ('x', float('nan')),
                             ('z', 0.0), ('observed', False)]:
            self.a.position_valid = True
            self.a.observed = True
            self.a.position_3d.x, self.a.position_3d.z = 0.0, 2.0
            if field in ('x', 'z'):
                setattr(self.a.position_3d, field, value)
            else:
                setattr(self.a, field, value)
            self.observe([self.a])
            self.collect()
            self.assertEqual(len(self.transforms), count)
        self.node.last_receive -= 1.0
        self.node.watchdog()
        self.collect()
        self.assertEqual(len(self.transforms), count)

    def test_select_enrolled_id_resolves_raw_track_and_rejects_raw_id(self):
        self.observe([self.a, self.b])
        self.assertFalse(self.select(8).success)
        self.assertTrue(self.select(2).success)
        self.assertEqual(self.node.selected, 8)
        self.assertEqual(self.node.identity, 2)
        self.assertTrue(self.select(3).success)
        self.assertEqual(self.node.selected, 9)
        self.assertEqual(self.node.identity, 3)

    def test_duplicate_enrolled_id_is_rejected_without_replacing_selection(self):
        self.observe([self.a, self.b])
        self.assertTrue(self.select(2).success)
        self.a.track_id = 3
        self.observe([self.a, self.b])
        self.assertFalse(self.select(3).success)
        self.assertEqual(self.node.identity, 2)
        self.assertFalse(self.select(1000000000).success)

    def test_configured_capture_age_does_not_relax_receipt_watchdog(self):
        self.node.destroy_node()
        self.node = module.FollowingTargetNode(parameter_overrides=[
            rclpy.parameter.Parameter('max_observation_age_sec', value=0.30)])
        msg = self.message([self.a])
        msg.header.stamp = rclpy.time.Time(
            nanoseconds=self.node.get_clock().now().nanoseconds - 225000000).to_msg()
        self.node.tracks(msg)
        self.node.recognized(msg)
        self.assertTrue(self.select(2).success)
        self.assertEqual(self.node.state, FollowingTarget.TRACKING)
        self.node.last_receive -= 0.16
        self.node.watchdog()
        self.assertEqual(self.node.state, FollowingTarget.LOST)
        self.assertFalse(self.select(2).success)

    def test_capture_age_over_configured_limit_is_rejected(self):
        self.node.destroy_node()
        self.node = module.FollowingTargetNode(parameter_overrides=[
            rclpy.parameter.Parameter('max_observation_age_sec', value=0.30)])
        msg = self.message([self.a])
        msg.header.stamp = rclpy.time.Time(
            nanoseconds=self.node.get_clock().now().nanoseconds - 350000000).to_msg()
        self.node.tracks(msg)
        self.node.recognized(msg)
        response = self.select(2)
        self.assertFalse(response.success)
        self.assertIn('maximum 300 ms', response.message)
