# Human identity tracking

Enrollment is explicit and starts only through the service, for a fresh observed,
unambiguous raw track:

```bash
ros2 service call /face_recognizer_node/enroll_person opl_interfaces/srv/EnrollPerson "{raw_track_id: 1, name: 'furia'}"
```

Keep the face visible while the authorized enrollment collects the configured
face samples. Authorization is cancelled if that trajectory disappears or becomes
ambiguous. No enrolled identity is created just by observing an unknown person.

After enrollment, an existing identity can be recognized automatically from three
consecutive distinct face observations. Body appearance alone cannot acquire or
transfer an enrolled ID. A continuous, unambiguous track retains its confirmed ID
when its face becomes unavailable. Crossings with indistinguishable associations
require face reacquisition; identical clothing cannot resolve complete overlap.

`TrackedHuman` exposes `observed`, `association_ambiguous`, `position_valid`, and
`identity_state`: `UNCONFIRMED`, `FACE_CONFIRMED`, `TRACKED`, or `UNCERTAIN`.
`match_score=0` on `TRACKED` means no fresh confirming face comparison, not an
unknown identity. Detector `confidence` retains its original meaning.

Person TF is disabled at startup. Select an enrolled person using the separate
following service using its enrolled `identity_id`. It must resolve to exactly one
freshly observed track with valid depth:

```bash
ros2 service call /select_following_target opl_interfaces/srv/SelectTarget "{identity_id: 1}"
```

Only that enrolled person's `opl_hr_person_<enrolled_id>` frame receives updates.
Cancel at any time, even without camera input:

```bash
ros2 service call /select_following_target opl_interfaces/srv/SelectTarget "{identity_id: 0}"
```

The selector is started by `test_detector.launch.py`. Selection stays bound to the
enrolled ID; another person cannot take over through a reused raw ID. TF stops on
loss, ambiguity, stale observations, or invalid depth. After loss or ambiguity,
a newer face confirmation of the same enrolled identity is required before updates
resume, including when that identity returns under a new raw ID. A temporary depth
failure alone does not require face reacquisition if tracking remains continuous.
`FollowingTarget.identity_id` identifies the requested person even while `LOST`;
only `valid=true` means its position is currently usable.

TF uses the camera optical frame and original observation timestamp, with a 150 ms
freshness limit. The quaternion is a frame convention, not measured body heading.
TF buffers retain historical samples after cancellation/loss; consumers must check
measurement age and target validity. This package does not command robot motion.

Tracker RGB is buffered for at most 30 frames and matched to the detection stamp
within 100 ms, including out-of-order arrivals. Missing RGB disables camera-motion
and appearance updates while geometric tracking continues. Valid positions require
matched RGB dimensions/frame metadata, registered depth, and valid calibration.
The RGB buffer clears on cleanup or a backward ROS clock jump.

`scrfd_conf_threshold` (default `0.45`) and `scrfd_nms_threshold` (default `0.30`)
are applied to SCRFD during configuration. Values must be finite and in `[0, 1]`;
parameter changes take effect after reconfiguration, not during active inference.

The optional webcam publisher reconnects at one-second intervals after open failure
or three consecutive failed reads (immediately detecting a closed device). It permits
five reconnect attempts, reapplies capture settings, and resets its retry budget only
after a frame is read successfully. Exhaustion requires restarting the node.

After rebuilding interfaces and the vision package, source this checkout's
`install/setup.bash` in every launch/consumer terminal and restart the old nodes.
Other overlays can still contain incompatible generated message definitions.

Verification: `colcon test --packages-select opl_human_vision`, then
`python3 -B -m unittest discover -s src/opl_human_vision/test -v`, from this workspace
with ROS and its local install overlay sourced. The recognizer-dropout test runs
the real node's lifecycle and subscriptions with GPU inference disabled; feature
and TF policy tests do not establish live model accuracy or Jetson performance.

## BoT-SORT comparison branch

`jetson_botsort` replaces the custom velocity/appearance-recovery tracker with a
C++ port of [BoT-SORT](https://github.com/NirAharon/BoT-SORT): an XYWH Kalman filter,
camera-motion compensation, joint IoU/OSNet association for strong detections,
IoU recovery from weak detections, tentative confirmation, and tracked/lost
duplicate removal. The pinned reference, MIT license, and OPL adaptations are in
[third_party/BoT-SORT](third_party/BoT-SORT/README.md).

Enrollment, recognition, enrolled-ID target selection, depth validity, and TF
freshness rules are unchanged. Ambiguous associations still require face
confirmation. A predicted box is not a real observation and cannot produce fresh
following TF. The default still hides boxes after one missed update; internal
retention remains 60 updates. Identical clothing cannot resolve a completely
ambiguous occlusion.

The launch now feeds detections down to `0.10`, with `0.65` for strong matching
and `0.70` for starting tracks. Weak detections cannot create tracks, confirm
initial tracks, or reacquire a lost track. Keep `detector_confidence_threshold`
no higher than `tracker_low_threshold` so weak detection recovery remains usable.

| Launch argument | Default | Meaning |
| --- | --- | --- |
| `tracker_low_threshold` | `0.10` | Weak detection floor |
| `tracker_high_threshold` | `0.65` | Strong detection minimum |
| `tracker_new_threshold` | `0.70` | New-track minimum |
| `tracker_match_threshold` | `0.80` | Maximum strong-match cost |
| `tracker_proximity_threshold` | `0.50` | Maximum IoU distance allowing appearance |
| `appearance_match_threshold` | `0.85` | Minimum body cosine similarity |

`tracker_iou_threshold`, `appearance_refresh_interval`, and the two
`appearance_spatial_*` launch arguments belong to the previous tracker and are
replaced by BoT-SORT's cost/proximity thresholds. OSNet now extracts each clean
strong detection on matched RGB frames and maintains a normalized feature EMA;
this costs more GPU work than the previous occasional extraction. Overlapping
crops, weak detections, and missing RGB cannot refresh that feature.

Save/commit the changes on `jetson_botsort` before switching, so they do not carry
over into the comparison branch. Stop the running vision pipeline before switching.
Rebuild and source
from the ROS workspace root after each switch; switching Git branches alone does
not replace the previously built shared libraries:

```bash
cd /home/furia/opl_ros
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-up-to opl_human_vision
source install/setup.bash
ros2 launch opl_human_vision test_detector.launch.py camera_type:=gemini335 enable_visualizer:=true
```

The tracker startup log identifies `BoT-SORT configured`. Use the same recorded
camera input and camera/model settings on both branches. Compare raw ID switches,
recovery after occlusion, false tracks, callback time, and FPS; recognition/TF
should still stop on ambiguity. Camera behavior and Jetson performance require
hardware testing; the deterministic tests do not establish either.

## Automatic MCP person following

The `following_target` node now exposes `/person_tracking/set_enabled` (`std_srvs/srv/SetBool`). Enabling requires exactly one fresh, unambiguous raw track with valid depth; no enrollment or identity ID is needed. It publishes that selected person's TF as `opl_hr_track_<raw_track_id>` and valid `/following_target` observations. Disabling clears the selection and stops TF updates. A selected target is never replaced automatically; ambiguity, disappearance, or a prolonged stream gap latches loss until the operator stops and starts following again. Temporary invalid depth pauses output and can recover on the same continuing raw track.

The existing `/select_following_target` enrolled-ID workflow remains available. The MCP `follow_person` tool automatically calls the tracking and following enable services, and `cancel_navigation` confirms following/navigation stop. The tracking node itself never commands base motion.

## Short-gap continuity

A selected confirmed identity survives a brief missed detection. Only the last
measured position may be reused within the configured capture-age budget (300 ms
in the deployed head-camera launch), with its original timestamp and the
existing receipt watchdog. No predicted position produces a new TF sample.
Ambiguity, conflicting/duplicate identities, prolonged loss and clock resets
clear the cache and require fresh identity confirmation.

The deployed on-demand system supervises the selector separately through
`human-vision-selector.service`, attached to `human-vision.service`. The parent
launch uses `launch_following_selector:=false` to avoid a duplicate selector.
This permits restarting the selector without clearing face-recognizer enrollment.
Standalone launches retain `launch_following_selector:=true` by default.

## Follower prediction eligibility

`FollowingTarget.observed` distinguishes actual current 3-D measurements from
cached sightings. `prediction_allowed` is true only for an ordinary missed
detection of the selected enrolled identity with fresh tracking input, a recent
last-good position, and no ambiguity or identity conflict. It does not authorize
raw-only following, invalid depth, expired identity evidence or pipeline failures.
The selector itself does not predict or move its measured person TF.

The robot-base `person_following` package uses those flags for a separately
marked estimate in odometry, lasting at most 2.5 s from the last capture. Its
predicted TF is `opl_hr_follow_target_predicted`; estimates cannot start following
or satisfy a new-sighting requirement. MCP startup now requires `observed=true`.

For the deployed selector, the `prediction.conf` systemd override sources
`26_OPL_HR/install_prediction_interfaces` after the normal install. This keeps
interface upgrades separate from the running recognizer's loaded libraries and
preserves its in-memory enrolled IDs. Source that overlay for standalone selector
launches too; all `FollowingTarget` consumers need the updated message definition.

The prediction window is now 2.5 s. Only an ordinary missed detection with a fresh
stream can retain the selected identity evidence for that window; ambiguity and
conflicting identities still stop immediately. `/person_tracking/prediction_status`
reports the exact eligibility gate, measured age and identity age for diagnosis.

## Vision control-panel registry

The recognizer exposes read-only `~/list_enrolled_identities` using
`opl_interfaces/srv/ListEnrolledIdentities`. It returns enrolled ID/name metadata
from the authoritative FaceEngine gallery under the existing engine mutex, plus
a session ID that changes when the gallery is recreated. Anonymous session IDs,
images and biometric templates are excluded. Inactive recognition returns an
unavailable response rather than an empty active registry. Robot Base Control's
Vision tab and the passive Vision MCP tools consume this metadata. Enrollment
remains in-memory and is lost on recognizer restart.
