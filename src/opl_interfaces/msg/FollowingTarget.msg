uint8 UNSELECTED=0
uint8 TRACKING=1
uint8 LOST=2
std_msgs/Header header
int32 raw_track_id
int32 identity_id
string name
geometry_msgs/Point position_3d
bool valid
uint8 state
# True only for a current valid measured 3-D observation; cached positions are false.
bool observed
# Only ordinary missed detections with fresh stream and unambiguous enrolled identity.
bool prediction_allowed
