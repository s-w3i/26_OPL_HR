#!/usr/bin/env python3
"""Service-selected person observations and TF; never command robot motion."""
from collections import Counter
from copy import deepcopy
import math
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from std_srvs.srv import SetBool
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from opl_interfaces.msg import FollowingTarget, TrackedHumanArray
from opl_interfaces.srv import SelectTarget


PREDICTION_HORIZON_SEC = 2.5


class FollowingTargetNode(Node):
    def __init__(self, **kwargs):
        super().__init__('following_target', **kwargs)
        self.declare_parameter('max_observation_age_sec', 0.15)
        self.max_observation_age_sec = self.get_parameter('max_observation_age_sec').value
        if not math.isfinite(self.max_observation_age_sec) or not 0 < self.max_observation_age_sec <= 0.75:
            raise ValueError('max_observation_age_sec must be finite and in (0, 0.75]')
        self.selected = 0
        self.automatic = False
        self.raw_counts = {}
        self.identity = 0
        self.name = ''
        self.visible = {}
        self.identities = {}
        self.uncertain = {}
        self.recognition_missing = set()
        self.identity_stamps = {}
        self.last_receive = 0.0
        self.last_header = None
        self.reacquire_after = -1
        self.last_tf = None
        self.last_good = None
        self.last_clock = self.get_clock().now().nanoseconds
        self.state = FollowingTarget.UNSELECTED
        self.pub = self.create_publisher(FollowingTarget, 'following_target', 1)
        self.prediction_status = self.create_publisher(DiagnosticArray, '/person_tracking/prediction_status', 1)
        self.broadcaster = TransformBroadcaster(self)
        self.create_subscription(TrackedHumanArray, 'tracked_humans', self.tracks, qos_profile_sensor_data)
        self.create_subscription(TrackedHumanArray, 'recognized_humans', self.recognized, qos_profile_sensor_data)
        self.create_service(SelectTarget, 'select_following_target', self.select)
        self.create_service(SetBool, 'person_tracking/set_enabled', self.auto_enable)
        self.create_timer(0.02, self.watchdog)

    @staticmethod
    def stamp(header):
        return header.stamp.sec * 10**9 + header.stamp.nanosec

    def age(self, header):
        return (self.get_clock().now().nanoseconds - self.stamp(header)) / 1e9

    def fresh(self):
        return (self.last_header is not None and time.monotonic() - self.last_receive <= 0.15
                and 0 <= self.age(self.last_header) <= self.max_observation_age_sec)

    def valid_observation(self, human):
        return (human is not None and human.observed and not human.association_ambiguous
                and human.position_valid and self.last_header is not None
                and bool(self.last_header.frame_id) and self.fresh()
                and all(math.isfinite(v) for v in (human.position_3d.x, human.position_3d.y, human.position_3d.z))
                and human.position_3d.z > 0)

    def lose(self, stamp):
        self.last_good = None
        if (self.identity or self.automatic) and self.reacquire_after < 0:
            self.reacquire_after = stamp

    def auto_enable(self, request, response):
        if not request.data:
            return self.select(SelectTarget.Request(identity_id=0), response)
        if self.automatic and self.selected and self.reacquire_after < 0:
            response.success, response.message = True, 'Current person remains selected'
            return response
        candidates = [raw for raw, human in self.visible.items()
                      if raw > 0 and self.raw_counts.get(raw) == 1 and self.valid_observation(human)]
        if len(candidates) != 1:
            response.success = False
            response.message = 'Exactly one fresh, unambiguous tracked person with valid depth is required'
            return response
        self.last_good = None
        self.selected, self.identity, self.name = candidates[0], 0, ''
        self.recognition_missing.clear()
        self.automatic = True
        self.uncertain.pop(self.selected, None)
        self.reacquire_after, self.last_tf = -1, None
        response.success, response.message = True, f'Target selected: opl_hr_track_{self.selected}'
        self.publish()
        return response

    def select(self, request, response):
        identity = request.identity_id
        if identity == 0:
            self.last_good = None
            self.recognition_missing.clear()
            self.selected, self.identity, self.name = 0, 0, ''
            self.automatic = False
            self.reacquire_after, self.last_tf = -1, None
            response.success, response.message = True, 'Target cleared; TF disabled'
            self.publish()
            return response
        now = self.get_clock().now().nanoseconds
        matches = [raw for raw, known in self.identities.items()
                   if known[0] == identity and raw in self.visible
                   and 0 <= (now - known[2]) / 1e9 <= 0.75]
        if not 0 < identity < 1000000000 or len(matches) != 1:
            response.success = False
            response.message = 'Enrolled ID must resolve to exactly one fresh visible track'
            return response
        raw = matches[0]
        known = self.identities[raw]
        if (not self.valid_observation(self.visible.get(raw)) or raw in self.uncertain or raw in self.recognition_missing
                or (self.reacquire_after >= 0 and identity == self.identity)):
            response.success = False
            human = self.visible[raw]
            response.message = (
                f'Target unavailable: observation age={self.age(self.last_header)*1000:.0f} ms '
                f'(maximum {self.max_observation_age_sec*1000:.0f} ms), observed={human.observed}, '
                f'depth_valid={human.position_valid}, ambiguous={human.association_ambiguous}, '
                f'identity_uncertain={raw in self.uncertain}, '
                f'face_reacquisition_required={self.reacquire_after >= 0 and identity == self.identity}')
            return response
        self.last_good = None
        self.selected = raw
        self.automatic = False
        self.identity, self.name = known[:2]
        self.reacquire_after = -1
        response.success = True
        response.message = f'Target selected: opl_hr_person_{self.identity}'
        self.publish()
        return response

    def recognized(self, msg):
        if self.automatic:
            return
        if not 0 <= self.age(msg.header) <= 0.75:
            return
        stamp = self.stamp(msg.header)
        candidates = []
        if self.identity and sum(h.observed and h.track_id == self.identity for h in msg.humans) > 1:
            self.lose(stamp)
            self.uncertain[self.selected] = stamp
        for human in msg.humans:
            raw = human.raw_track_id
            current = self.visible.get(raw)
            if raw <= 0 or current is None or stamp < self.identity_stamps.get(raw, -1):
                continue
            self.identity_stamps[raw] = stamp
            enrolled = (human.observed and not human.association_ambiguous
                        and 0 < human.track_id < 1000000000
                        and human.identity_state in (human.FACE_CONFIRMED, human.TRACKED))
            if human.association_ambiguous or human.identity_state == human.UNCERTAIN:
                self.uncertain[raw] = max(stamp, self.uncertain.get(raw, -1))
            if not enrolled:
                if human.observed:
                    known = self.identities.get(raw)
                    ordinary_recognition_gap = (
                        raw == self.selected and self.identity > 0 and known is not None
                        and known[0] == self.identity and human.identity_state == human.UNCONFIRMED
                        and not human.association_ambiguous and raw not in self.uncertain
                        and self.reacquire_after < 0 and self.last_good is not None
                        and self.last_good[2] == (self.identity, raw)
                        and 0 <= self.age(self.last_good[0]) < PREDICTION_HORIZON_SEC)
                    if ordinary_recognition_gap:
                        self.recognition_missing.add(raw)
                    else:
                        self.identities.pop(raw, None)
                        if raw == self.selected:
                            self.lose(stamp)
                continue
            self.recognition_missing.discard(raw)
            if raw in self.uncertain:
                if human.identity_state != human.FACE_CONFIRMED or stamp <= self.uncertain[raw]:
                    continue
                del self.uncertain[raw]
            self.identities[raw] = (human.track_id, human.name, stamp)
            if raw == self.selected and human.track_id != self.identity:
                self.lose(stamp)
            if (self.identity and human.track_id == self.identity
                    and human.identity_state == human.FACE_CONFIRMED
                    and stamp > self.reacquire_after and current.observed
                    and not current.association_ambiguous):
                candidates.append(raw)
        if self.reacquire_after >= 0 and len(candidates) == 1 and self.fresh():
            self.selected = candidates[0]
            self.reacquire_after = -1
        self.publish()

    def tracks(self, msg):
        if self.last_header is not None and self.stamp(msg.header) <= self.stamp(self.last_header):
            return
        # A brief pose timeout pauses output; only a prolonged stream gap loses continuity.
        if (self.identity or self.automatic) and time.monotonic() - self.last_receive > 0.75:
            self.lose(self.get_clock().now().nanoseconds)
        self.last_header = msg.header
        self.last_receive = time.monotonic()
        self.visible = {h.raw_track_id: h for h in msg.humans}
        self.raw_counts = Counter(h.raw_track_id for h in msg.humans)
        stamp = self.stamp(msg.header)
        for human in msg.humans:
            if human.association_ambiguous:
                self.uncertain[human.raw_track_id] = max(stamp, self.uncertain.get(human.raw_track_id, -1))
                self.identities.pop(human.raw_track_id, None)
        current = self.visible.get(self.selected)
        ambiguous = (current is not None and current.association_ambiguous) or self.raw_counts.get(self.selected, 0) > 1
        missing = current is None or not current.observed or self.selected in self.recognition_missing
        loss_horizon = PREDICTION_HORIZON_SEC if self.identity and not self.automatic else .75
        if ambiguous or (missing and (self.last_good is None or self.age(self.last_good[0]) >= loss_horizon)):
            self.lose(stamp)
        self.uncertain = {k: v for k, v in self.uncertain.items() if k in self.visible or k == self.selected}
        self.identities = {k: v for k, v in self.identities.items() if k in self.visible or k == self.selected}
        self.identity_stamps = {k: v for k, v in self.identity_stamps.items() if k in self.visible or k == self.selected}
        self.publish()

    def publish(self):
        out = FollowingTarget()
        if self.last_header is not None:
            out.header = self.last_header
        out.raw_track_id, out.identity_id, out.name = self.selected, self.identity, self.name
        out.state = FollowingTarget.LOST if (self.identity or self.automatic) else FollowingTarget.UNSELECTED
        human = self.visible.get(self.selected)
        known = self.identities.get(self.selected)
        identity_age = ((self.get_clock().now().nanoseconds-known[2])/1e9 if known else float('inf'))
        measured_age = self.age(self.last_good[0]) if self.last_good else float('inf')
        missing = human is None or not human.observed or self.selected in self.recognition_missing
        ordinary_window = bool(self.identity and missing and self.fresh()
            and self.reacquire_after < 0 and self.selected not in self.uncertain
            and self.last_good is not None and self.last_good[2] == (self.identity,self.selected)
            and 0 <= measured_age < PREDICTION_HORIZON_SEC)
        known_fresh = known is not None and 0 <= identity_age <= (
            PREDICTION_HORIZON_SEC if ordinary_window else .75)
        if self.identity and (not known_fresh or known[0] != self.identity):
            self.lose(self.stamp(out.header))
        if ((self.identity or self.automatic) and self.reacquire_after < 0 and (self.automatic or known_fresh)
                and self.selected not in self.uncertain):
            if self.selected not in self.recognition_missing and self.valid_observation(human):
                self.last_good = (deepcopy(self.last_header), deepcopy(human), (self.identity, self.selected))
            elif (self.last_good is not None and self.last_good[2] == (self.identity, self.selected)
                  and 0 <= self.age(self.last_good[0]) <= self.max_observation_age_sec
                  and time.monotonic() - self.last_receive <= .15):
                out.header, human = self.last_good[:2]
            else:
                human = None
        else:
            human = None
        current = self.visible.get(self.selected)
        out.observed = human is not None and self.selected not in self.recognition_missing and self.valid_observation(current)
        missing = current is None or not current.observed or self.selected in self.recognition_missing
        gates = {
            'no_enrolled_selection': self.identity > 0 and not self.automatic,
            'not_an_ordinary_missed_detection': missing,
            'tracking_stream_stale': self.fresh(),
            'depth_unavailable': current is None or not current.observed or self.valid_observation(current),
            'ambiguous_association': self.selected not in self.uncertain and
                self.raw_counts.get(self.selected,0) <= 1 and
                (current is None or not current.association_ambiguous),
            'identity_evidence_missing_or_conflicting': known is not None and known[0] == self.identity,
            'identity_evidence_expired': known_fresh,
            'identity_reacquisition_required': self.reacquire_after < 0,
            'no_measured_history': self.last_good is not None and self.last_good[2] == (self.identity,self.selected),
            'prediction_capture_deadline': 0 <= measured_age < PREDICTION_HORIZON_SEC,
        }
        out.prediction_allowed = all(gates.values())
        reason = 'measured' if out.observed else 'eligible' if out.prediction_allowed else next(
            reason for reason,ok in gates.items() if not ok)
        diagnostics = DiagnosticArray(status=[DiagnosticStatus(
            name='following_prediction_eligibility', message=reason,
            values=[KeyValue(key='identity_id',value=str(self.identity)),
                    KeyValue(key='raw_track_id',value=str(self.selected)),
                    KeyValue(key='measurement_age',value=str(measured_age)),
                    KeyValue(key='identity_age',value=str(identity_age)),
                    KeyValue(key='recognition_confirmation_missing',value=str(self.selected in self.recognition_missing).lower())])])
        diagnostics.header.stamp = self.get_clock().now().to_msg()
        self.prediction_status.publish(diagnostics)
        if human is not None:
            out.position_3d = human.position_3d
            out.valid, out.state = True, FollowingTarget.TRACKING
            frame = f'opl_hr_track_{self.selected}' if self.automatic else f'opl_hr_person_{self.identity}'
            key = (frame, self.stamp(out.header))
            if key != self.last_tf:
                transform = TransformStamped()
                transform.header = out.header
                transform.child_frame_id = frame
                transform.transform.translation.x = human.position_3d.x
                transform.transform.translation.y = human.position_3d.y
                transform.transform.translation.z = human.position_3d.z
                transform.transform.rotation.w = 1.0
                self.broadcaster.sendTransform(transform)
                self.last_tf = key
        self.state = out.state
        self.pub.publish(out)

    def watchdog(self):
        now = self.get_clock().now().nanoseconds
        if now < self.last_clock:
            self.lose(now)
            self.reacquire_after = now
            self.last_header = None
            self.visible.clear()
            self.identities.clear()
            self.recognition_missing.clear()
            self.identity_stamps.clear()
            self.publish()
        self.last_clock = now
        if (self.identity or self.automatic) and not self.fresh():
            if time.monotonic() - self.last_receive > 0.75:
                self.lose(self.get_clock().now().nanoseconds)
            self.publish()


def main():
    rclpy.init()
    node = FollowingTargetNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
