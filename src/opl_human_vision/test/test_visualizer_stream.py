import base64
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import rclpy
from opl_interfaces.msg import TrackedHuman, TrackedHumanArray

spec = importlib.util.spec_from_file_location('human_visualizer',
    Path(__file__).parents[1]/'scripts/visualizer.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


@pytest.fixture
def visualizer():
    assert os.environ['ROS_DOMAIN_ID'] == '231'
    rclpy.init(args=['--ros-args', '-p', 'show_window:=false', '-p', 'stream_jpeg:=true'])
    node = module.PipelineVisualizer()
    yield node
    node.destroy_node(); rclpy.shutdown()


def render(node, human_age=0., image_age=0.):
    image = node.bridge.cv2_to_imgmsg(np.zeros((480, 640, 3), dtype=np.uint8), encoding='bgr8')
    stamp = node.get_clock().now().nanoseconds
    image.header.stamp = rclpy.time.Time(nanoseconds=stamp-int(image_age*1e9)).to_msg()
    humans = TrackedHumanArray()
    humans.header.stamp = rclpy.time.Time(nanoseconds=stamp-int(human_age*1e9)).to_msg()
    person = TrackedHuman(track_id=7, raw_track_id=1, name='Furia', observed=True, confidence=.9)
    person.bbox.x_offset = 20; person.bbox.y_offset = 80
    person.bbox.width = 150; person.bbox.height = 250; person.position_3d.z = 1.5
    humans.humans = [person]
    node.recognized_callback(humans)
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream), patch.object(cv2, 'imshow') as show:
        node.image_callback(image); show.assert_not_called()
    assert not node.bridge.imgmsg_to_cv2(image).any(), 'visualizer must not alter its source image'
    return stream.getvalue()


def decode(line):
    assert line.startswith('VISION_FRAME ')
    packet = json.loads(line[13:])
    return cv2.imdecode(np.frombuffer(base64.b64decode(packet['jpeg']), dtype=np.uint8), cv2.IMREAD_COLOR)


def test_stream_contains_annotations_without_opening_a_window(visualizer):
    assert decode(render(visualizer)).any()


def test_stale_recognition_does_not_draw_cached_boxes(visualizer):
    assert not decode(render(visualizer, human_age=1.)).any()


def test_stale_images_are_not_streamed(visualizer):
    assert render(visualizer, image_age=2.) == ''
