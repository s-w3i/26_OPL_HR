"""Automatic raw-track selection; no camera or robot motion."""
import importlib.util
from pathlib import Path
import time
import unittest

import rclpy
from opl_interfaces.msg import TrackedHuman, TrackedHumanArray, FollowingTarget
from std_srvs.srv import SetBool

spec = importlib.util.spec_from_file_location('auto_target_test', Path(__file__).parents[1]/'scripts/following_target.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class AutomaticFollowingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = module.FollowingTargetNode()

    def tearDown(self):
        self.node.destroy_node()

    def person(self, raw):
        person = TrackedHuman(raw_track_id=raw, observed=True, position_valid=True)
        person.position_3d.z = 2.0
        return person

    def observe(self, people):
        message = TrackedHumanArray(humans=people)
        message.header.frame_id = 'head_camera_color_optical_frame'
        message.header.stamp = self.node.get_clock().now().to_msg()
        self.node.tracks(message)
        return message

    def enable(self, value=True):
        return self.node.auto_enable(SetBool.Request(data=value), SetBool.Response())

    def test_one_unenrolled_person_can_be_selected_and_disabled(self):
        self.observe([self.person(8)])
        self.assertTrue(self.enable().success)
        self.assertEqual((self.node.identity, self.node.selected), (0, 8))
        self.assertEqual(self.node.state, FollowingTarget.TRACKING)
        self.assertTrue(self.enable(False).success)
        self.assertEqual(self.node.state, FollowingTarget.UNSELECTED)

    def test_zero_multiple_ambiguous_stale_and_duplicate_tracks_are_rejected(self):
        for people in ([], [self.person(1), self.person(2)], [self.person(1), self.person(1)]):
            self.observe(people)
            self.assertFalse(self.enable().success)
        person = self.person(1)
        person.association_ambiguous = True
        self.observe([person])
        self.assertFalse(self.enable().success)
        self.observe([self.person(1)])
        self.node.last_receive -= 1.0
        self.assertFalse(self.enable().success)

    def test_keeps_selected_person_when_another_person_enters(self):
        self.observe([self.person(1)])
        self.assertTrue(self.enable().success)
        self.observe([self.person(1), self.person(2)])
        self.assertTrue(self.enable().success)
        self.assertEqual(self.node.selected, 1)
        self.assertEqual(self.node.state, FollowingTarget.TRACKING)

    def test_lost_or_ambiguous_person_is_never_replaced_automatically(self):
        self.observe([self.person(1)])
        self.enable()
        self.node.last_good[0].stamp.sec -= 1
        self.observe([self.person(2)])
        self.assertEqual(self.node.selected, 1)
        self.assertEqual(self.node.state, FollowingTarget.LOST)
        self.observe([self.person(1), self.person(2)])
        self.assertEqual(self.node.state, FollowingTarget.LOST)
        self.assertFalse(self.enable().success)
        self.enable(False)
        self.observe([self.person(2)])
        self.assertTrue(self.enable().success)
        self.assertEqual(self.node.selected, 2)

    def test_depth_pause_recovers_without_face_recognition(self):
        person = self.person(1)
        self.observe([person])
        self.enable()
        person.position_valid = False
        message = self.observe([person])
        self.assertEqual(self.node.state, FollowingTarget.TRACKING)
        self.node.last_good[0].stamp.sec -= 1
        message = self.observe([person])
        self.assertEqual(self.node.state, FollowingTarget.LOST)
        self.node.recognized(message)
        person.position_valid = True
        self.observe([person])
        self.assertEqual(self.node.state, FollowingTarget.TRACKING)


if __name__ == '__main__':
    unittest.main()
