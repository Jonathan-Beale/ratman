import bpy
import os
import sys
import json
import math
import numpy as np
from pathlib import Path
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector

OUTPUT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "Renders"))
os.makedirs(OUTPUT_ROOT, exist_ok=True)
try:
    from PIL import Image, ImageDraw
    PIL_AVAILABLE = True
    print("Pillow import OK")
except ImportError as e:
    PIL_AVAILABLE = False
    print(f"Pillow import failed: {e}")
    Image = None
    ImageDraw = None


argv = sys.argv
if "--" not in argv:
    raise RuntimeError("Missing -- arguments")

args = argv[argv.index("--") + 1:]

# Strip optional named args: --width W  --height H  --max-cameras N
RENDER_WIDTH  = 1920
RENDER_HEIGHT = 1080
MAX_CAMERAS   = None  # None = all cameras
_filtered = []
_i = 0
while _i < len(args):
    if args[_i] == "--width" and _i + 1 < len(args):
        RENDER_WIDTH = int(args[_i + 1]); _i += 2
    elif args[_i] == "--height" and _i + 1 < len(args):
        RENDER_HEIGHT = int(args[_i + 1]); _i += 2
    elif args[_i] == "--max-cameras" and _i + 1 < len(args):
        MAX_CAMERAS = int(args[_i + 1]); _i += 2
    else:
        _filtered.append(args[_i]); _i += 1
args = _filtered

if len(args) < 4:
    raise RuntimeError("Usage: -- [--width W] [--height H] [--max-cameras N] <output_root> <num_actors> <model1> <anim1> [<model2> <anim2> ...]")

OUTPUT_ROOT = args[0]
NUM_ACTORS = int(args[1])
if len(args) < 2 + NUM_ACTORS * 2:
    raise RuntimeError(f"Expected {NUM_ACTORS} model+anim pairs but got {(len(args) - 2)} args after num_actors")
ACTORS = [(args[2 + i * 2], args[2 + i * 2 + 1]) for i in range(NUM_ACTORS)]

# Backward-compat aliases used by logging and single-export metadata
MODEL_FILE = ACTORS[0][0]
ANIM_FILE  = ACTORS[0][1]

ACTOR_SPACING    = 1.5   # world-space metres between actors on X axis (fallback when no spawn points)
FLOOR_SNAP_MARGIN = 0.0  # set to zero so mesh sits exactly on the ground
                          # accounts for render-time modifier evaluation producing slightly deeper
                          # geometry than the pre-render scan (corrective smooth, subsurf, etc.)

# Names of spawn point objects in the .blend scene, ordered by actor index.
SPAWN_POINT_NAMES = ["SpawnPoint_Primary", "SpawnPoint_Secondary"]

scene = bpy.context.scene


# =========================
# Canonical joint definition
# =========================

CANONICAL_JOINTS = [
    "pelvis",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "spine",
    "thorax",
    "neck",
    "head",
    # Face bones (added for pose estimation style)
    "nose",
    "eye.L",
    "eye.R",
    "ear.L",
    "ear.R",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    # Left hand fingers
    "lefthandthumb1", "lefthandthumb2", "lefthandthumb3",
    "lefthandindex1", "lefthandindex2", "lefthandindex3",
    "lefthandmiddle1", "lefthandmiddle2", "lefthandmiddle3",
    "lefthandring1", "lefthandring2", "lefthandring3",
    "lefthandpinky1", "lefthandpinky2", "lefthandpinky3",
    # Right hand fingers
    "righthandthumb1", "righthandthumb2", "righthandthumb3",
    "righthandindex1", "righthandindex2", "righthandindex3",
    "righthandmiddle1", "righthandmiddle2", "righthandmiddle3",
    "righthandring1", "righthandring2", "righthandring3",
    "righthandpinky1", "righthandpinky2", "righthandpinky3",
]

# =========================================
# H36M 17-joint layout (VideoPose3D format)
# =========================================
# Standard Human3.6M ordering used by VideoPose3D.
H36M_JOINTS = [
    "pelvis",         # 0
    "right_hip",      # 1
    "right_knee",     # 2
    "right_ankle",    # 3
    "left_hip",       # 4
    "left_knee",      # 5
    "left_ankle",     # 6
    "spine",          # 7
    "thorax",         # 8
    "neck",           # 9
    "head",           # 10
    "left_shoulder",  # 11
    "left_elbow",     # 12
    "left_wrist",     # 13
    "right_shoulder", # 14
    "right_elbow",    # 15
    "right_wrist",    # 16
]

H36M_LEFT_JOINTS  = [4, 5, 6, 11, 12, 13]
H36M_RIGHT_JOINTS = [1, 2, 3, 14, 15, 16]

# =========================================
# COCO layout (DWPose format)
# =========================================
# Body: 17 joints (COCO keypoints). None = not in our rig (will be NaN).
COCO_BODY_JOINTS = [
    "nose",           # 0: nose
    "eye.L",          # 1: left_eye
    "eye.R",          # 2: right_eye
    "ear.L",          # 3: left_ear
    "ear.R",          # 4: right_ear
    "left_shoulder",  # 5
    "right_shoulder", # 6
    "left_elbow",     # 7
    "right_elbow",    # 8
    "left_wrist",     # 9
    "right_wrist",    # 10
    "left_hip",       # 11
    "right_hip",      # 12
    "left_knee",      # 13
    "right_knee",     # 14
    "left_ankle",     # 15
    "right_ankle",    # 16
]

# Hand: 21 joints per hand (COCO wholebody / DWPose).
# None = tip bones (segment 4) which we don't track.
COCO_HAND_LEFT_JOINTS = [
    "left_wrist",      # 0: wrist
    "lefthandthumb1",  # 1: thumb CMC
    "lefthandthumb2",  # 2: thumb MCP
    "lefthandthumb3",  # 3: thumb IP
    None,              # 4: thumb tip
    "lefthandindex1",  # 5: index MCP
    "lefthandindex2",  # 6: index PIP
    "lefthandindex3",  # 7: index DIP
    None,              # 8: index tip
    "lefthandmiddle1", # 9: middle MCP
    "lefthandmiddle2", # 10: middle PIP
    "lefthandmiddle3", # 11: middle DIP
    None,              # 12: middle tip
    "lefthandring1",   # 13: ring MCP
    "lefthandring2",   # 14: ring PIP
    "lefthandring3",   # 15: ring DIP
    None,              # 16: ring tip
    "lefthandpinky1",  # 17: pinky MCP
    "lefthandpinky2",  # 18: pinky PIP
    "lefthandpinky3",  # 19: pinky DIP
    None,              # 20: pinky tip
]

COCO_HAND_RIGHT_JOINTS = [
    "right_wrist",      # 0: wrist
    "righthandthumb1",  # 1
    "righthandthumb2",  # 2
    "righthandthumb3",  # 3
    None,               # 4: thumb tip
    "righthandindex1",  # 5
    "righthandindex2",  # 6
    "righthandindex3",  # 7
    None,               # 8: index tip
    "righthandmiddle1", # 9
    "righthandmiddle2", # 10
    "righthandmiddle3", # 11
    None,               # 12: middle tip
    "righthandring1",   # 13
    "righthandring2",   # 14
    "righthandring3",   # 15
    None,               # 16: ring tip
    "righthandpinky1",  # 17
    "righthandpinky2",  # 18
    "righthandpinky3",  # 19
    None,               # 20: pinky tip
]

# =========================================
# DWPose / OpenPose body_18 joint order
# =========================================
# Maps each of the 18 OpenPose body joints to a canonical joint name in our rig.
# None = joint not available from the rig (marked absent in the subset array).
# Neck (index 1) is computed as the midpoint of left_shoulder and right_shoulder.
DWPOSE_BODY18_JOINTS = [
    "nose",           # 0: nose
    None,             # 1: neck       (computed as shoulder midpoint — handled specially)
    "right_shoulder", # 2
    "right_elbow",    # 3
    "right_wrist",    # 4
    "left_shoulder",  # 5
    "left_elbow",     # 6
    "left_wrist",     # 7
    "right_hip",      # 8
    "right_knee",     # 9
    "right_ankle",    # 10
    "left_hip",       # 11
    "left_knee",      # 12
    "left_ankle",     # 13
    "eye.R",         # 14: right_eye
    "eye.L",         # 15: left_eye
    "ear.R",         # 16: right_ear
    "ear.L",         # 17: left_ear
]

RETARGET_BONE_MAP = {
    # Hands / wrists
    "lefthand": [
        "mixamorig:LeftHand", "LeftHand", "hand.L", "hand_l", "l_hand", "lefthand"
    ],
    "righthand": [
        "mixamorig:RightHand", "RightHand", "hand.R", "hand_r", "r_hand", "righthand"
    ],

    # Thumb
    "lefthandthumb1": [
        "mixamorig:LeftHandThumb1", "LeftHandThumb1", "thumb.01.L", "thumb1.L",
        "thumb_01_l", "l_thumb1", "lefthandthumb1"
    ],
    "lefthandthumb2": [
        "mixamorig:LeftHandThumb2", "LeftHandThumb2", "thumb.02.L", "thumb2.L",
        "thumb_02_l", "l_thumb2", "lefthandthumb2"
    ],
    "lefthandthumb3": [
        "mixamorig:LeftHandThumb3", "LeftHandThumb3", "thumb.03.L", "thumb3.L",
        "thumb_03_l", "l_thumb3", "lefthandthumb3"
    ],
    "righthandthumb1": [
        "mixamorig:RightHandThumb1", "RightHandThumb1", "thumb.01.R", "thumb1.R",
        "thumb_01_r", "r_thumb1", "righthandthumb1"
    ],
    "righthandthumb2": [
        "mixamorig:RightHandThumb2", "RightHandThumb2", "thumb.02.R", "thumb2.R",
        "thumb_02_r", "r_thumb2", "righthandthumb2"
    ],
    "righthandthumb3": [
        "mixamorig:RightHandThumb3", "RightHandThumb3", "thumb.03.R", "thumb3.R",
        "thumb_03_r", "r_thumb3", "righthandthumb3"
    ],

    # Index
    "lefthandindex1": [
        "mixamorig:LeftHandIndex1", "LeftHandIndex1", "f_index.01.L", "index.01.L",
        "index1.L", "index_01_l", "l_index1", "lefthandindex1"
    ],
    "lefthandindex2": [
        "mixamorig:LeftHandIndex2", "LeftHandIndex2", "f_index.02.L", "index.02.L",
        "index2.L", "index_02_l", "l_index2", "lefthandindex2"
    ],
    "lefthandindex3": [
        "mixamorig:LeftHandIndex3", "LeftHandIndex3", "f_index.03.L", "index.03.L",
        "index3.L", "index_03_l", "l_index3", "lefthandindex3"
    ],
    "righthandindex1": [
        "mixamorig:RightHandIndex1", "RightHandIndex1", "f_index.01.R", "index.01.R",
        "index1.R", "index_01_r", "r_index1", "righthandindex1"
    ],
    "righthandindex2": [
        "mixamorig:RightHandIndex2", "RightHandIndex2", "f_index.02.R", "index.02.R",
        "index2.R", "index_02_r", "r_index2", "righthandindex2"
    ],
    "righthandindex3": [
        "mixamorig:RightHandIndex3", "RightHandIndex3", "f_index.03.R", "index.03.R",
        "index3.R", "index_03_r", "r_index3", "righthandindex3"
    ],

    # Middle
    "lefthandmiddle1": [
        "mixamorig:LeftHandMiddle1", "LeftHandMiddle1", "f_middle.01.L", "middle.01.L",
        "middle1.L", "middle_01_l", "l_middle1", "lefthandmiddle1"
    ],
    "lefthandmiddle2": [
        "mixamorig:LeftHandMiddle2", "LeftHandMiddle2", "f_middle.02.L", "middle.02.L",
        "middle2.L", "middle_02_l", "l_middle2", "lefthandmiddle2"
    ],
    "lefthandmiddle3": [
        "mixamorig:LeftHandMiddle3", "LeftHandMiddle3", "f_middle.03.L", "middle.03.L",
        "middle3.L", "middle_03_l", "l_middle3", "lefthandmiddle3"
    ],
    "righthandmiddle1": [
        "mixamorig:RightHandMiddle1", "RightHandMiddle1", "f_middle.01.R", "middle.01.R",
        "middle1.R", "middle_01_r", "r_middle1", "righthandmiddle1"
    ],
    "righthandmiddle2": [
        "mixamorig:RightHandMiddle2", "RightHandMiddle2", "f_middle.02.R", "middle.02.R",
        "middle2.R", "middle_02_r", "r_middle2", "righthandmiddle2"
    ],
    "righthandmiddle3": [
        "mixamorig:RightHandMiddle3", "RightHandMiddle3", "f_middle.03.R", "middle.03.R",
        "middle3.R", "middle_03_r", "r_middle3", "righthandmiddle3"
    ],

    # Ring
    "lefthandring1": [
        "mixamorig:LeftHandRing1", "LeftHandRing1", "f_ring.01.L", "ring.01.L",
        "ring1.L", "ring_01_l", "l_ring1", "lefthandring1"
    ],
    "lefthandring2": [
        "mixamorig:LeftHandRing2", "LeftHandRing2", "f_ring.02.L", "ring.02.L",
        "ring2.L", "ring_02_l", "l_ring2", "lefthandring2"
    ],
    "lefthandring3": [
        "mixamorig:LeftHandRing3", "LeftHandRing3", "f_ring.03.L", "ring.03.L",
        "ring3.L", "ring_03_l", "l_ring3", "lefthandring3"
    ],
    "righthandring1": [
        "mixamorig:RightHandRing1", "RightHandRing1", "f_ring.01.R", "ring.01.R",
        "ring1.R", "ring_01_r", "r_ring1", "righthandring1"
    ],
    "righthandring2": [
        "mixamorig:RightHandRing2", "RightHandRing2", "f_ring.02.R", "ring.02.R",
        "ring2.R", "ring_02_r", "r_ring2", "righthandring2"
    ],
    "righthandring3": [
        "mixamorig:RightHandRing3", "RightHandRing3", "f_ring.03.R", "ring.03.R",
        "ring3.R", "ring_03_r", "r_ring3", "righthandring3"
    ],

    # Pinky
    "lefthandpinky1": [
        "mixamorig:LeftHandPinky1", "LeftHandPinky1", "f_pinky.01.L", "pinky.01.L",
        "pinky1.L", "pinky_01_l", "l_pinky1", "lefthandpinky1"
    ],
    "lefthandpinky2": [
        "mixamorig:LeftHandPinky2", "LeftHandPinky2", "f_pinky.02.L", "pinky.02.L",
        "pinky2.L", "pinky_02_l", "l_pinky2", "lefthandpinky2"
    ],
    "lefthandpinky3": [
        "mixamorig:LeftHandPinky3", "LeftHandPinky3", "f_pinky.03.L", "pinky.03.L",
        "pinky3.L", "pinky_03_l", "l_pinky3", "lefthandpinky3"
    ],
    "righthandpinky1": [
        "mixamorig:RightHandPinky1", "RightHandPinky1", "f_pinky.01.R", "pinky.01.R",
        "pinky1.R", "pinky_01_r", "r_pinky1", "righthandpinky1"
    ],
    "righthandpinky2": [
        "mixamorig:RightHandPinky2", "RightHandPinky2", "f_pinky.02.R", "pinky.02.R",
        "pinky2.R", "pinky_02_r", "r_pinky2", "righthandpinky2"
    ],
    "righthandpinky3": [
        "mixamorig:RightHandPinky3", "RightHandPinky3", "f_pinky.03.R", "pinky.03.R",
        "pinky3.R", "pinky_03_r", "r_pinky3", "righthandpinky3"
    ],
}

# Tweak these aliases to match your rigs.
# Order matters: first match wins.
    # ...existing code...
BONE_NAME_MAP = {
    # Face bones (add aliases as needed for your rigs)
    "nose": [
        "nose", "Nose"
    ],
    "eye.L": [
        "eye.L", "eye_l", "lefteye", "left_eye", "Eye.L", "Eye_L"
    ],
    "eye.R": [
        "eye.R", "eye_r", "righteye", "right_eye", "Eye.R", "Eye_R"
    ],
    "ear.L": [
        "ear.L", "ear_l", "leftear", "left_ear", "Ear.L", "Ear_L"
    ],
    "ear.R": [
        "ear.R", "ear_r", "rightear", "right_ear", "Ear.R", "Ear_R"
    ],
    "pelvis": [
        "mixamorig:Hips", "Hips", "pelvis", "hips", "hip", "root",
    ],
    "right_hip": [
        "rightupleg", "thigh.r", "upperleg.r", "r_thigh", "r_upleg",
        "mixamorig:RightUpLeg", "RightUpLeg"
    ],
    "right_knee": [
        "rightleg", "shin.r", "lowerleg.r", "r_calf", "r_leg",
        "mixamorig:RightLeg", "RightLeg"
    ],
    "right_ankle": [
        "rightfoot", "foot.r", "r_foot",
        "mixamorig:RightFoot", "RightFoot"
    ],
    "left_hip": [
        "leftupleg", "thigh.l", "upperleg.l", "l_thigh", "l_upleg",
        "mixamorig:LeftUpLeg", "LeftUpLeg"
    ],
    "left_knee": [
        "leftleg", "shin.l", "lowerleg.l", "l_calf", "l_leg",
        "mixamorig:LeftLeg", "LeftLeg"
    ],
    "left_ankle": [
        "leftfoot", "foot.l", "l_foot",
        "mixamorig:LeftFoot", "LeftFoot"
    ],
    "spine": [
        "spine", "spine1", "spine_01", "mixamorig:Spine1", "Spine"
    ],
    "thorax": [
        "mixamorig:Neck", "spine2", "spine3", "chest", "upperchest", "spine_02", "spine_03",
        "mixamorig:Spine2", "Chest", "UpperChest"
    ],
    "neck": [
        "mixamorig:Neck", "neck", "mixamorig:Neck", "Neck"
    ],
    "head": [
        "head", "mixamorig:Head", "Head"
    ],
    "left_shoulder": [
        "leftarm", "upperarm.l", "l_upperarm", "l_arm",
        "mixamorig:LeftArm", "LeftArm"
    ],
    "left_elbow": [
        "leftforearm", "forearm.l", "lowerarm.l", "l_forearm",
        "mixamorig:LeftForeArm", "LeftForeArm"
    ],
    "left_wrist": [
        "lefthand", "hand.l", "l_hand",
        "mixamorig:LeftHand", "LeftHand"
    ],
    "right_shoulder": [
        "rightarm", "upperarm.r", "r_upperarm", "r_arm",
        "mixamorig:RightArm", "RightArm"
    ],
    "right_elbow": [
        "rightforearm", "forearm.r", "lowerarm.r", "r_forearm",
        "mixamorig:RightForeArm", "RightForeArm"
    ],
    "right_wrist": [
        "righthand", "hand.r", "r_hand",
        "mixamorig:RightHand", "RightHand"
    ],
    # Left hand fingers (1, 2, 3 segments)
    "lefthandthumb1": [
        "mixamorig:LeftHandThumb1", "LeftHandThumb1", "thumb.01.L", "thumb1.L",
        "thumb_01_l", "l_thumb1", "lefthandthumb1"
    ],
    "lefthandthumb2": [
        "mixamorig:LeftHandThumb2", "LeftHandThumb2", "thumb.02.L", "thumb2.L",
        "thumb_02_l", "l_thumb2", "lefthandthumb2"
    ],
    "lefthandthumb3": [
        "mixamorig:LeftHandThumb3", "LeftHandThumb3", "thumb.03.L", "thumb3.L",
        "thumb_03_l", "l_thumb3", "lefthandthumb3"
    ],
    "lefthandindex1": [
        "mixamorig:LeftHandIndex1", "LeftHandIndex1", "f_index.01.L", "index.01.L",
        "index1.L", "index_01_l", "l_index1", "lefthandindex1"
    ],
    "lefthandindex2": [
        "mixamorig:LeftHandIndex2", "LeftHandIndex2", "f_index.02.L", "index.02.L",
        "index2.L", "index_02_l", "l_index2", "lefthandindex2"
    ],
    "lefthandindex3": [
        "mixamorig:LeftHandIndex3", "LeftHandIndex3", "f_index.03.L", "index.03.L",
        "index3.L", "index_03_l", "l_index3", "lefthandindex3"
    ],
    "lefthandmiddle1": [
        "mixamorig:LeftHandMiddle1", "LeftHandMiddle1", "f_middle.01.L", "middle.01.L",
        "middle1.L", "middle_01_l", "l_middle1", "lefthandmiddle1"
    ],
    "lefthandmiddle2": [
        "mixamorig:LeftHandMiddle2", "LeftHandMiddle2", "f_middle.02.L", "middle.02.L",
        "middle2.L", "middle_02_l", "l_middle2", "lefthandmiddle2"
    ],
    "lefthandmiddle3": [
        "mixamorig:LeftHandMiddle3", "LeftHandMiddle3", "f_middle.03.L", "middle.03.L",
        "middle3.L", "middle_03_l", "l_middle3", "lefthandmiddle3"
    ],
    "lefthandring1": [
        "mixamorig:LeftHandRing1", "LeftHandRing1", "f_ring.01.L", "ring.01.L",
        "ring1.L", "ring_01_l", "l_ring1", "lefthandring1"
    ],
    "lefthandring2": [
        "mixamorig:LeftHandRing2", "LeftHandRing2", "f_ring.02.L", "ring.02.L",
        "ring2.L", "ring_02_l", "l_ring2", "lefthandring2"
    ],
    "lefthandring3": [
        "mixamorig:LeftHandRing3", "LeftHandRing3", "f_ring.03.L", "ring.03.L",
        "ring3.L", "ring_03_l", "l_ring3", "lefthandring3"
    ],
    "lefthandpinky1": [
        "mixamorig:LeftHandPinky1", "LeftHandPinky1", "f_pinky.01.L", "pinky.01.L",
        "pinky1.L", "pinky_01_l", "l_pinky1", "lefthandpinky1"
    ],
    "lefthandpinky2": [
        "mixamorig:LeftHandPinky2", "LeftHandPinky2", "f_pinky.02.L", "pinky.02.L",
        "pinky2.L", "pinky_02_l", "l_pinky2", "lefthandpinky2"
    ],
    "lefthandpinky3": [
        "mixamorig:LeftHandPinky3", "LeftHandPinky3", "f_pinky.03.L", "pinky.03.L",
        "pinky3.L", "pinky_03_l", "l_pinky3", "lefthandpinky3"
    ],
    # Right hand fingers (1, 2, 3 segments)
    "righthandthumb1": [
        "mixamorig:RightHandThumb1", "RightHandThumb1", "thumb.01.R", "thumb1.R",
        "thumb_01_r", "r_thumb1", "righthandthumb1"
    ],
    "righthandthumb2": [
        "mixamorig:RightHandThumb2", "RightHandThumb2", "thumb.02.R", "thumb2.R",
        "thumb_02_r", "r_thumb2", "righthandthumb2"
    ],
    "righthandthumb3": [
        "mixamorig:RightHandThumb3", "RightHandThumb3", "thumb.03.R", "thumb3.R",
        "thumb_03_r", "r_thumb3", "righthandthumb3"
    ],
    "righthandindex1": [
        "mixamorig:RightHandIndex1", "RightHandIndex1", "f_index.01.R", "index.01.R",
        "index1.R", "index_01_r", "r_index1", "righthandindex1"
    ],
    "righthandindex2": [
        "mixamorig:RightHandIndex2", "RightHandIndex2", "f_index.02.R", "index.02.R",
        "index2.R", "index_02_r", "r_index2", "righthandindex2"
    ],
    "righthandindex3": [
        "mixamorig:RightHandIndex3", "RightHandIndex3", "f_index.03.R", "index.03.R",
        "index3.R", "index_03_r", "r_index3", "righthandindex3"
    ],
    "righthandmiddle1": [
        "mixamorig:RightHandMiddle1", "RightHandMiddle1", "f_middle.01.R", "middle.01.R",
        "middle1.R", "middle_01_r", "r_middle1", "righthandmiddle1"
    ],
    "righthandmiddle2": [
        "mixamorig:RightHandMiddle2", "RightHandMiddle2", "f_middle.02.R", "middle.02.R",
        "middle2.R", "middle_02_r", "r_middle2", "righthandmiddle2"
    ],
    "righthandmiddle3": [
        "mixamorig:RightHandMiddle3", "RightHandMiddle3", "f_middle.03.R", "middle.03.R",
        "middle3.R", "middle_03_r", "r_middle3", "righthandmiddle3"
    ],
    "righthandring1": [
        "mixamorig:RightHandRing1", "RightHandRing1", "f_ring.01.R", "ring.01.R",
        "ring1.R", "ring_01_r", "r_ring1", "righthandring1"
    ],
    "righthandring2": [
        "mixamorig:RightHandRing2", "RightHandRing2", "f_ring.02.R", "ring.02.R",
        "ring2.R", "ring_02_r", "r_ring2", "righthandring2"
    ],
    "righthandring3": [
        "mixamorig:RightHandRing3", "RightHandRing3", "f_ring.03.R", "ring.03.R",
        "ring3.R", "ring_03_r", "r_ring3", "righthandring3"
    ],
    "righthandpinky1": [
        "mixamorig:RightHandPinky1", "RightHandPinky1", "f_pinky.01.R", "pinky.01.R",
        "pinky1.R", "pinky_01_r", "r_pinky1", "righthandpinky1"
    ],
    "righthandpinky2": [
        "mixamorig:RightHandPinky2", "RightHandPinky2", "f_pinky.02.R", "pinky.02.R",
        "pinky2.R", "pinky_02_r", "r_pinky2", "righthandpinky2"
    ],
    "righthandpinky3": [
        "mixamorig:RightHandPinky3", "RightHandPinky3", "f_pinky.03.R", "pinky.03.R",
        "pinky3.R", "pinky_03_r", "r_pinky3", "righthandpinky3"
    ],
    # Thumb 4
    "lefthandthumb4": [
        "mixamorig:LeftHandThumb4", "LeftHandThumb4", "thumb.04.L", "thumb4.L",
        "thumb_04_l", "l_thumb4", "lefthandthumb4", "thumb_end.L", "l_thumb_end"
    ],
    "righthandthumb4": [
        "mixamorig:RightHandThumb4", "RightHandThumb4", "thumb.04.R", "thumb4.R",
        "thumb_04_r", "r_thumb4", "righthandthumb4", "thumb_end.R", "r_thumb_end"
    ],

    # Middle 4
    "lefthandmiddle4": [
        "mixamorig:LeftHandMiddle4", "LeftHandMiddle4", "f_middle.04.L", "middle.04.L",
        "middle4.L", "middle_04_l", "l_middle4", "lefthandmiddle4", "middle_end.L"
    ],
    "righthandmiddle4": [
        "mixamorig:RightHandMiddle4", "RightHandMiddle4", "f_middle.04.R", "middle.04.R",
        "middle4.R", "middle_04_r", "r_middle4", "righthandmiddle4", "middle_end.R"
    ],

    # Ring 4
    "lefthandring4": [
        "mixamorig:LeftHandRing4", "LeftHandRing4", "f_ring.04.L", "ring.04.L",
        "ring4.L", "ring_04_l", "l_ring4", "lefthandring4", "ring_end.L"
    ],
    "righthandring4": [
        "mixamorig:RightHandRing4", "RightHandRing4", "f_ring.04.R", "ring.04.R",
        "ring4.R", "ring_04_r", "r_ring4", "righthandring4", "ring_end.R"
    ],

    # Pinky 4
    "lefthandpinky4": [
        "mixamorig:LeftHandPinky4", "LeftHandPinky4", "f_pinky.04.L", "pinky.04.L",
        "pinky4.L", "pinky_04_l", "l_pinky4", "lefthandpinky4", "pinky_end.L"
    ],
    "righthandpinky4": [
        "mixamorig:RightHandPinky4", "RightHandPinky4", "f_pinky.04.R", "pinky.04.R",
        "pinky4.R", "pinky_04_r", "r_pinky4", "righthandpinky4", "pinky_end.R"
    ],
}

# DWPose-ish colored body graph
SKELETON_EDGES = [
    ("pelvis", "right_hip"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("pelvis", "left_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("pelvis", "spine"),
    ("spine", "thorax"),
    ("thorax", "neck"),
    # Face bones connections (if present)
    ("neck", "nose"),
    ("nose", "eye.L"),
    ("nose", "eye.R"),
    ("eye.L", "ear.L"),
    ("eye.R", "ear.R"),
    ("head", "ear.L"),
    ("head", "ear.R"),
    ("thorax", "left_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("thorax", "right_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    # Left hand fingers
    ("left_wrist", "lefthandthumb1"), ("lefthandthumb1", "lefthandthumb2"), ("lefthandthumb2", "lefthandthumb3"),
    ("left_wrist", "lefthandindex1"), ("lefthandindex1", "lefthandindex2"), ("lefthandindex2", "lefthandindex3"),
    ("left_wrist", "lefthandmiddle1"), ("lefthandmiddle1", "lefthandmiddle2"), ("lefthandmiddle2", "lefthandmiddle3"),
    ("left_wrist", "lefthandring1"), ("lefthandring1", "lefthandring2"), ("lefthandring2", "lefthandring3"),
    ("left_wrist", "lefthandpinky1"), ("lefthandpinky1", "lefthandpinky2"), ("lefthandpinky2", "lefthandpinky3"),
    # Right hand fingers
    ("right_wrist", "righthandthumb1"), ("righthandthumb1", "righthandthumb2"), ("righthandthumb2", "righthandthumb3"),
    ("right_wrist", "righthandindex1"), ("righthandindex1", "righthandindex2"), ("righthandindex2", "righthandindex3"),
    ("right_wrist", "righthandmiddle1"), ("righthandmiddle1", "righthandmiddle2"), ("righthandmiddle2", "righthandmiddle3"),
    ("right_wrist", "righthandring1"), ("righthandring1", "righthandring2"), ("righthandring2", "righthandring3"),
    ("right_wrist", "righthandpinky1"), ("righthandpinky1", "righthandpinky2"), ("righthandpinky2", "righthandpinky3"),
]

EDGE_COLORS = {
    ("pelvis", "right_hip"): (255, 80, 80),
    ("right_hip", "right_knee"): (255, 110, 110),
    ("right_knee", "right_ankle"): (255, 140, 140),
    ("pelvis", "left_hip"): (80, 160, 255),
    ("left_hip", "left_knee"): (110, 180, 255),
    ("left_knee", "left_ankle"): (140, 200, 255),
    ("pelvis", "spine"): (255, 220, 80),
    ("spine", "thorax"): (255, 220, 80),
    ("thorax", "neck"): (255, 220, 80),
    ("neck", "nose"): (255, 220, 80),
    ("thorax", "left_shoulder"): (80, 255, 140),
    ("left_shoulder", "left_elbow"): (100, 255, 160),
    ("left_elbow", "left_wrist"): (120, 255, 180),
    ("thorax", "right_shoulder"): (255, 120, 255),
    ("right_shoulder", "right_elbow"): (255, 150, 255),
    ("right_elbow", "right_wrist"): (255, 180, 255),
}

# Per-limb body colors for DWPose rendering.
# 18 entries matching limb_seq order (see _dwpose_draw_body):
#   [2,3],[2,6],[3,4],[4,5],[6,7],[7,8],[2,9],[9,10],[10,11],
#   [2,12],[12,13],[13,14],[2,1],[1,15],[15,17],[1,16],[16,18],[3,17]
#
# Model A: right-side = Red shades, left-side = Green shades, center = Gold.
# Model B: right-side = Blue shades, left-side = Orange shades, center = Gold.
# Darker = proximal, lighter = distal.
MODEL_A_BODY_LIMB_COLORS = [
    (155,   0,   0),  #  0 [2,3]  neck→R_shoulder      R darkest
    (  0, 130,   0),  #  1 [2,6]  neck→L_shoulder      L darkest
    (210,  25,  25),  #  2 [3,4]  R_shoulder→R_elbow   R dark
    (255, 105, 105),  #  3 [4,5]  R_elbow→R_wrist      R bright
    ( 15, 205,  15),  #  4 [6,7]  L_shoulder→L_elbow   L mid
    ( 85, 255,  85),  #  5 [7,8]  L_elbow→L_wrist      L bright
    (175,  10,  10),  #  6 [2,9]  neck→R_hip            R dark
    (230,  55,  55),  #  7 [9,10] R_hip→R_knee         R mid
    (255, 135, 135),  #  8 [10,11]R_knee→R_ankle       R lightest
    (  0, 150,   0),  #  9 [2,12] neck→L_hip            L dark
    ( 35, 220,  35),  # 10 [12,13]L_hip→L_knee         L mid
    (115, 255, 115),  # 11 [13,14]L_knee→L_ankle       L lightest
    (255, 200,  50),  # 12 [2,1]  neck→head             center gold
    (205,  25,  25),  # 13 [1,15] head→R_eye            R med
    (245,  90,  90),  # 14 [15,17]R_eye→R_ear           R light
    ( 15, 190,  15),  # 15 [1,16] head→L_eye            L med
    ( 65, 245,  65),  # 16 [16,18]L_eye→L_ear           L light
    (220,  40,  40),  # 17 [3,17] R_shoulder→R_ear      R med
]

MODEL_B_BODY_LIMB_COLORS = [
    (  0,   0, 150),  #  0 [2,3]  neck→R_shoulder      B darkest
    (185,  75,   0),  #  1 [2,6]  neck→L_shoulder      O darkest
    ( 25,  25, 195),  #  2 [3,4]  R_shoulder→R_elbow   B dark
    (120, 120, 255),  #  3 [4,5]  R_elbow→R_wrist      B bright
    (225, 120,  10),  #  4 [6,7]  L_shoulder→L_elbow   O mid
    (255, 200, 110),  #  5 [7,8]  L_elbow→L_wrist      O bright
    (  0,   5, 160),  #  6 [2,9]  neck→R_hip            B dark
    ( 55,  55, 220),  #  7 [9,10] R_hip→R_knee         B mid
    (115, 115, 255),  #  8 [10,11]R_knee→R_ankle       B brightest
    (200,  90,   0),  #  9 [2,12] neck→L_hip            O dark
    (240, 145,  30),  # 10 [12,13]L_hip→L_knee         O mid
    (255, 195, 105),  # 11 [13,14]L_knee→L_ankle       O brightest
    (255, 200,  50),  # 12 [2,1]  neck→head             center gold
    ( 20,  20, 205),  # 13 [1,15] head→R_eye            B med
    ( 90,  90, 248),  # 14 [15,17]R_eye→R_ear           B light
    (220, 110,  10),  # 15 [1,16] head→L_eye            O med
    (250, 175,  75),  # 16 [16,18]L_eye→L_ear           O light
    ( 45,  45, 210),  # 17 [3,17] R_shoulder→R_ear      B med
]

# Per-hand base colors: [left_hand_color, right_hand_color]
# Used to tint the hand skeleton; edges are shaded dark (knuckle) → light (fingertip).
MODEL_A_HAND_COLORS = [(  0, 200,   0), (200,   0,   0)]  # [left=green, right=red]
MODEL_B_HAND_COLORS = [(230, 130,  20), (  0,   0, 220)]  # [left=orange, right=blue]

# Standard MusePose/OpenPose rainbow palette — 18 limbs cycling through the spectrum.
# Matches util.draw_bodypose exactly so output is compatible with stock MusePose inference.
RAINBOW_BODY_LIMB_COLORS = [
    (255,   0,   0), (255,  85,   0), (255, 170,   0), (255, 255,   0),
    (170, 255,   0), ( 85, 255,   0), (  0, 255,   0), (  0, 255,  85),
    (  0, 255, 170), (  0, 255, 255), (  0, 170, 255), (  0,  85, 255),
    (  0,   0, 255), ( 85,   0, 255), (170,   0, 255), (255,   0, 255),
    (255,   0, 170),
]
RAINBOW_HAND_COLORS = [(  0, 255,   0), (255,   0,   0)]  # left=green, right=red

# Indexed by actor (actor 0 = Model A, actor 1 = Model B)
ACTOR_BODY_LIMB_COLORS = [MODEL_A_BODY_LIMB_COLORS, MODEL_B_BODY_LIMB_COLORS]
ACTOR_HAND_COLORS      = [MODEL_A_HAND_COLORS,      MODEL_B_HAND_COLORS]

# Maps each of the 18 body joint indices (0-indexed, DWPOSE_BODY18_JOINTS order) to the
# limb color index (into MODEL_*_BODY_LIMB_COLORS) that best represents that joint.
# Right-side joints → red limb palette; left-side → green; center → gold (limb 12).
JOINT_TO_LIMB_COLOR = [
    12,  #  0  nose/head      → gold (center)
    12,  #  1  neck (computed)→ gold (center)
     0,  #  2  R_shoulder     → limb 0 (R darkest)
     2,  #  3  R_elbow        → limb 2 (R dark)
     3,  #  4  R_wrist        → limb 3 (R bright)
     1,  #  5  L_shoulder     → limb 1 (L darkest)
     4,  #  6  L_elbow        → limb 4 (L mid)
     5,  #  7  L_wrist        → limb 5 (L bright)
     6,  #  8  R_hip          → limb 6 (R dark)
     7,  #  9  R_knee         → limb 7 (R mid)
     8,  # 10  R_ankle        → limb 8 (R lightest)
     9,  # 11  L_hip          → limb 9 (L dark)
    10,  # 12  L_knee         → limb 10 (L mid)
    11,  # 13  L_ankle        → limb 11 (L lightest)
    13,  # 14  R_eye          → limb 13 (R med)
    15,  # 15  L_eye          → limb 15 (L med)
    14,  # 16  R_ear          → limb 14 (R light)
    16,  # 17  L_ear          → limb 16 (L light)
]


def normalize_name(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "")
        .replace("-", "")
        .replace("_", "")
    )


def import_file(path: str):
    ext = Path(path).suffix.lower()
    before_objs = {obj.name for obj in bpy.data.objects}

    if ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    elif ext in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == ".bvh":
        bpy.ops.import_anim.bvh(filepath=path)
    elif ext == ".blend":
        with bpy.data.libraries.load(path, link=False) as (data_from, data_to):
            data_to.objects = data_from.objects
        for obj in data_to.objects:
            if obj is not None:
                bpy.context.collection.objects.link(obj)
    else:
        raise RuntimeError(f"Unsupported file type: {path}")

    new_objs = [obj for obj in bpy.data.objects if obj.name not in before_objs]
    return new_objs


def add_corrective_smooth(model_objs):
    """
    Add a Corrective Smooth modifier AFTER the Armature modifier on every mesh in
    the model.  This reduces sharp deformation artifacts such as hood/cloth warping
    during head rotation without requiring any manual weight-paint edits.

    The modifier executes at render time and does not affect bone baking.
    """
    for obj in model_objs:
        if obj.type != "MESH":
            continue
        # Skip if already present (re-import guard)
        if any(m.type == "CORRECTIVE_SMOOTH" for m in obj.modifiers):
            continue
        mod = obj.modifiers.new(name="CorrectiveSmooth", type="CORRECTIVE_SMOOTH")
        mod.factor = 0.5
        mod.iterations = 5
        mod.smooth_type = "LENGTH_WEIGHTED"
        mod.use_pin_boundary = True  # keeps boundary edges from shrinking inward
        print(f"  CorrectiveSmooth added to '{obj.name}'")


def add_shoulder_limits(target_armature):
    """
    Add Limit Rotation constraints to both upper-arm bones before baking.
    Prevents extreme retargeted rotations from driving the arm mesh inside the body.

    Because bake_to_target() uses visual_keying=True + clear_constraints=True,
    these limits are baked into the keyframes and then removed automatically —
    no manual cleanup needed.
    """
    limit_rad = math.radians(150.0)
    for canonical in ("left_shoulder", "right_shoulder"):
        pb = resolve_first_matching_bone(target_armature, BONE_NAME_MAP[canonical])
        if pb is None:
            print(f"  Shoulder limit: '{canonical}' not found on target rig, skipping")
            continue
        c = pb.constraints.new(type="LIMIT_ROTATION")
        c.name = "AutoLimitRot"
        c.owner_space = "LOCAL"
        c.use_limit_x = True;  c.min_x = -limit_rad;  c.max_x = limit_rad
        c.use_limit_y = True;  c.min_y = -limit_rad;  c.max_y = limit_rad
        c.use_limit_z = True;  c.min_z = -limit_rad;  c.max_z = limit_rad
        print(f"  Shoulder limit ±150° added to '{pb.name}'")


def print_hand_related_bones(label, armature_obj):
    print(f"\n{label} hand-related bones:")
    for pb in armature_obj.pose.bones:
        n = pb.name.lower()
        if any(tok in n for tok in ["hand", "thumb", "index", "middle", "ring", "pinky", "finger"]):
            print(f"  {pb.name}")


def find_first_armature(objects):
    arms = [obj for obj in objects if obj.type == "ARMATURE"]
    if not arms:
        raise RuntimeError("No armature found in imported objects")
    return arms[0]


def get_scene_cameras():
    cams = [obj for obj in scene.objects if obj.type == "CAMERA"]
    if not cams:
        raise RuntimeError("No cameras found in the loaded .blend scene")
    return cams


def get_spawn_locations():
    """
    Return spawn point objects from the scene in actor order.
    Looks for SPAWN_POINT_NAMES in order; skips names not present.
    """
    spawns = []
    for name in SPAWN_POINT_NAMES:
        obj = scene.objects.get(name)
        if obj is not None:
            spawns.append(obj)
    return spawns


def build_pose_bone_lookup(armature_obj):
    lookup = {}
    for pb in armature_obj.pose.bones:
        lookup[normalize_name(pb.name)] = pb
    return lookup


def resolve_first_matching_bone(armature_obj, aliases):
    lookup = build_pose_bone_lookup(armature_obj)
    for alias in aliases:
        pb = lookup.get(normalize_name(alias))
        if pb is not None:
            return pb
    return None


def resolve_canonical_bones(armature_obj):
    resolved = {}
    missing = []

    # First pass: try to resolve all canonical joints
    for joint_name in CANONICAL_JOINTS:
        aliases = BONE_NAME_MAP[joint_name]
        pb = resolve_first_matching_bone(armature_obj, aliases)
        if pb is None:
            missing.append(joint_name)
        else:
            resolved[joint_name] = pb

    # Special fallback: if all index finger bones are present but other fingers are missing, map missing fingers (except thumb) to index finger bones
    left_index = [f"lefthandindex{i}" for i in (1,2,3)]
    right_index = [f"righthandindex{i}" for i in (1,2,3)]
    left_index_found = all(j in resolved for j in left_index)
    right_index_found = all(j in resolved for j in right_index)

    # Map left hand fingers
    if left_index_found:
        for finger in ["middle", "ring", "pinky"]:
            for i in (1,2,3):
                canon = f"lefthand{finger}{i}"
                if canon not in resolved:
                    resolved[canon] = resolved[f"lefthandindex{i}"]
                    if canon in missing:
                        missing.remove(canon)

    # Map right hand fingers
    if right_index_found:
        for finger in ["middle", "ring", "pinky"]:
            for i in (1,2,3):
                canon = f"righthand{finger}{i}"
                if canon not in resolved:
                    resolved[canon] = resolved[f"righthandindex{i}"]
                    if canon in missing:
                        missing.remove(canon)

    # Allow missing thumb bones: only raise error if missing non-thumb canonical joints
    non_thumb_missing = [j for j in missing if not (
        j.startswith("lefthandthumb") or j.startswith("righthandthumb")
    )]
    if non_thumb_missing:
        raise RuntimeError(
            "Missing canonical joints on target armature: "
            + ", ".join(non_thumb_missing)
            + "\nAvailable bones:\n"
            + ", ".join(pb.name for pb in armature_obj.pose.bones)
        )

    # For missing thumb bones, just skip them (do not add to resolved)
    return resolved


def create_constraints(source_armature, target_armature):
    """
    Retarget target rig bones from source rig bones.
    Uses BONE_NAME_MAP canonical alias resolution and armature-space pose matrix
    retargeting to correctly handle skeletons with different rest poses.
    """
    frame_start, frame_end = get_frame_range(source_armature)

    # Build bone_map: source_bone_name -> target_bone_name
    # Uses BONE_NAME_MAP alias resolution so rigs with different naming conventions
    # (e.g. BVH "LeftArm" vs Mixamo "mixamorig:LeftArm") are matched correctly.
    bone_map = {}
    for _canonical_name, aliases in BONE_NAME_MAP.items():
        src_pb = resolve_first_matching_bone(source_armature, aliases)
        tgt_pb = resolve_first_matching_bone(target_armature, aliases)
        if src_pb and tgt_pb:
            bone_map[src_pb.name] = tgt_pb.name

    if not bone_map:
        raise RuntimeError(
            "No matching bones found between source and target armatures. "
            "Check BONE_NAME_MAP aliases.\n"
            f"Source bones: {[b.name for b in source_armature.pose.bones]}\n"
            f"Target bones: {[b.name for b in target_armature.pose.bones]}"
        )

    print(f"[Retarget] Mapped {len(bone_map)} bones")

    # Sort bone pairs root-to-tip so that when we set a child bone's matrix,
    # its parent's matrix has already been updated for this frame.
    def bone_depth(armature, name):
        pb = armature.pose.bones.get(name)
        depth = 0
        while pb and pb.parent:
            depth += 1
            pb = pb.parent
        return depth

    sorted_pairs = sorted(bone_map.items(), key=lambda kv: bone_depth(target_armature, kv[1]))

    # Pre-compute the armature-space basis change.
    # This corrects for any world-placement difference between the two armatures
    # (e.g. one at origin rotated 90°, the other not).
    armature_basis = target_armature.matrix_world.inverted() @ source_armature.matrix_world

    # Helper: get root bone world-space head position
    def get_root_world_head(armature, root_bone_name):
        root_bone = armature.pose.bones.get(root_bone_name)
        if root_bone is not None:
            return armature.matrix_world @ root_bone.head
        return None

    # Determine root bone names using the same normalized lookup used for bone_map,
    # so the search is case-insensitive and returns the bone's actual name.
    # We keep separate names for source and target because the two rigs may name
    # their root bone differently (e.g. BVH "Hips" vs Mixamo "mixamorig:Hips").
    _src_lookup = build_pose_bone_lookup(source_armature)
    _tgt_lookup = build_pose_bone_lookup(target_armature)
    _root_candidates = ["pelvis", "hips", "root"]

    _src_root_pb = None
    for _c in _root_candidates:
        _src_root_pb = _src_lookup.get(_c)
        if _src_root_pb:
            break
    src_root_bone_name = _src_root_pb.name if _src_root_pb else next(iter(source_armature.pose.bones)).name

    # Prefer the bone_map entry (already resolved via BONE_NAME_MAP aliases).
    # Fall back to the same candidate search on the target armature.
    tgt_root_bone_name = bone_map.get(src_root_bone_name)
    if tgt_root_bone_name is None:
        for _c in _root_candidates:
            _tgt_root_pb = _tgt_lookup.get(_c)
            if _tgt_root_pb:
                tgt_root_bone_name = _tgt_root_pb.name
                break
    if tgt_root_bone_name is None:
        tgt_root_bone_name = next(iter(target_armature.pose.bones)).name

    # Log world coordinates per frame
    target_bone_log_path = os.path.join(OUTPUT_ROOT, "target_bone_world_coords.txt")
    source_bone_log_path = os.path.join(OUTPUT_ROOT, "source_bone_world_coords.txt")
    target_bone_names = [pb.name for pb in target_armature.pose.bones]
    source_bone_names = [pb.name for pb in source_armature.pose.bones]

    # Compute the target root bone's rest position as a world-space offset from the
    # armature origin.  This is constant for the whole animation (bones don't scale,
    # and we never move the armature during retargeting until we key it ourselves).
    # We use it to invert the bone's own rest height when placing the armature, so
    # that setting:
    #   target_armature.location = src_root_world - tgt_root_rest_world_offset
    # guarantees:
    #   world_hips = armature.location + tgt_root_rest_world_offset = src_root_world
    #
    # A delta-based approach (previous) only works when source and target start at the
    # same elevation.  For fall animations that begin mid-air, the large negative delta
    # drags the armature below Z=0 and the mesh clips through the floor.
    bpy.context.scene.frame_set(frame_start)
    tgt_root_bone_obj = target_armature.pose.bones.get(tgt_root_bone_name)
    if tgt_root_bone_obj is not None:
        # World-space rest offset of the root bone from the armature origin.
        # Used to compute the armature XY position so world_hip_xy == src_root_xy.
        tgt_root_rest_world_offset = (
            target_armature.matrix_world.to_3x3()
            @ tgt_root_bone_obj.bone.matrix_local.translation
        )
        # Precompute a matrix that maps world-space displacement to bone local space.
        # bone.matrix_local is in armature space; multiplying by matrix_world.to_3x3()
        # brings it to world space.  Inverting lets us map a world-Z delta to the
        # corresponding bone-local displacement, regardless of how the bone is oriented.
        tgt_root_world_to_local = (
            target_armature.matrix_world.to_3x3()
            @ Matrix(tgt_root_bone_obj.bone.matrix_local).to_3x3()
        ).inverted()
    else:
        tgt_root_rest_world_offset = Vector((0.0, 0.0, 0.0))
        tgt_root_world_to_local = Matrix.Identity(3)

    # Record the armature's initial scene Z — this never changes, so any camera
    # that tracks or is parented to the armature object stays at a stable height.
    tgt_armature_init_z = target_armature.location.z

    with open(target_bone_log_path, "w") as target_log, open(source_bone_log_path, "w") as source_log:
        target_log.write("frame,bone,x,y,z\n")
        source_log.write("frame,bone,x,y,z\n")

        for frame in range(frame_start, frame_end + 1):
            bpy.context.scene.frame_set(frame)

            # Retarget rotations processing root-to-tip.
            #
            # We copy only the ROTATION from src_bone.matrix (armature space), not the
            # translation.  Copying the full 4x4 matrix would force the target bone's
            # head to match the source's world position, distorting the mesh when the
            # two skeletons have different bone lengths or proportions.
            #
            # Derivation (Blender armature-space formula, rotation-only 3x3 sub-matrices):
            #   arm_rot = parent_arm_rot @ R @ local_rot
            #   where R = (parent.bone.matrix_local.inverted() @ bone.matrix_local).to_3x3()
            #   => local_rot = R.inverted() @ parent_arm_rot.inverted() @ desired_arm_rot
            #
            # view_layer.update() is called after each bone so the next child bone reads
            # an up-to-date parent.matrix when computing its own local rotation.
            for src_name, tgt_name in sorted_pairs:
                src_bone = source_armature.pose.bones.get(src_name)
                tgt_bone = target_armature.pose.bones.get(tgt_name)
                if not src_bone or not tgt_bone:
                    continue

                tgt_bone.rotation_mode = 'QUATERNION'

                # Desired armature-space rotation (rotation part of source pose matrix,
                # converted to target armature space via armature_basis).
                desired_arm_rot = (armature_basis @ src_bone.matrix).to_3x3()

                if tgt_bone.parent:
                    # R = rest rotation of this bone relative to its parent, in armature space
                    R = (tgt_bone.parent.bone.matrix_local.inverted() @ tgt_bone.bone.matrix_local).to_3x3()
                    parent_arm_rot = tgt_bone.parent.matrix.to_3x3()
                else:
                    R = Matrix(tgt_bone.bone.matrix_local).to_3x3()
                    parent_arm_rot = Matrix.Identity(3)

                local_rot = R.inverted() @ parent_arm_rot.inverted() @ desired_arm_rot
                tgt_bone.rotation_quaternion = local_rot.to_4x4().to_quaternion()

                # Propagate so the next child bone reads an up-to-date parent.matrix
                bpy.context.view_layer.update()
                tgt_bone.keyframe_insert(data_path="rotation_quaternion", frame=frame)

            # Root motion: drive the armature object with the source's world-space delta.
            #
            # We do NOT copy the root bone's local position directly.  Doing so would
            # apply the source skeleton's absolute hip height as a local offset on top
            # of the target skeleton's own rest height, double-applying the elevation
            # and causing floor penetration during low-to-ground moves like rolls.
            #
            # Instead, we measure the source root's world-space movement relative to
            # frame_start and add that delta to the target armature's initial scene
            # location.  Because the target's hip height is already encoded in the bone's
            # rest matrix (armature space), the character stays correctly elevated:
            #   world_hips = armature.location.z + hip_rest_in_armature_space
            #
            # The root bone local location is explicitly zeroed so it never drifts from
            # its rest position; all locomotion comes from the armature object.
            src_root_world = get_root_world_head(source_armature, src_root_bone_name)

            # Armature object: XY locomotion only.  Z is locked to the initial scene
            # height so cameras parented/tracking the armature don't fall with the character.
            if src_root_world is not None:
                target_armature.location.x = src_root_world.x - tgt_root_rest_world_offset.x
                target_armature.location.y = src_root_world.y - tgt_root_rest_world_offset.y
                target_armature.location.z = tgt_armature_init_z
                target_armature.keyframe_insert(data_path="location", frame=frame)

            # Root bone local displacement: carries the vertical offset so cameras stay stable.
            # The desired world-Z of the hip is src_root_world.z.
            # The armature object is locked at tgt_armature_init_z, and the bone's rest
            # world-Z contribution is tgt_root_rest_world_offset.z.
            # We compute the world-space delta and rotate it into bone local space,
            # so the displacement is correct regardless of bone orientation (fixes crawling float).
            tgt_root = target_armature.pose.bones.get(tgt_root_bone_name)
            if tgt_root is not None:
                if src_root_world is not None:
                    delta_world_z = src_root_world.z - (tgt_armature_init_z + tgt_root_rest_world_offset.z)
                    bone_local_disp = tgt_root_world_to_local @ Vector((0.0, 0.0, delta_world_z))
                else:
                    bone_local_disp = Vector((0.0, 0.0, 0.0))
                tgt_root.location = bone_local_disp
                tgt_root.keyframe_insert(data_path="location", frame=frame)

            # Log world coordinates of all bones for this frame
            for bone_name in target_bone_names:
                pb = target_armature.pose.bones.get(bone_name)
                if pb is not None:
                    world_head = target_armature.matrix_world @ pb.head
                    target_log.write(f"{frame},{bone_name},{world_head.x},{world_head.y},{world_head.z}\n")
            for bone_name in source_bone_names:
                pb = source_armature.pose.bones.get(bone_name)
                if pb is not None:
                    world_head = source_armature.matrix_world @ pb.head
                    source_log.write(f"{frame},{bone_name},{world_head.x},{world_head.y},{world_head.z}\n")


def get_frame_range(source_armature):
    if source_armature.animation_data and source_armature.animation_data.action:
        start, end = source_armature.animation_data.action.frame_range
        return int(start), int(end)
    return 1, 250


def bake_to_target(target_armature, frame_start, frame_end):
    bpy.ops.object.select_all(action="DESELECT")
    target_armature.select_set(True)
    bpy.context.view_layer.objects.active = target_armature

    bpy.ops.nla.bake(
        frame_start=frame_start,
        frame_end=frame_end,
        only_selected=False,
        visual_keying=True,
        clear_constraints=True,
        use_current_action=True,
        bake_types={'POSE'},
    )


def get_render_dimensions(scene):
    scale = scene.render.resolution_percentage / 100.0
    width = int(scene.render.resolution_x * scale)
    height = int(scene.render.resolution_y * scale)
    return width, height


FACE_JOINTS_USE_TAIL = {"nose", "eye.L", "eye.R", "ear.L", "ear.R"}

def bone_world_head(armature_obj, pose_bone, joint_name):
    # Face joints: use the tail (tip) of the bone — the root sits at the head pivot,
    # the tail points to the actual facial landmark position.
    if joint_name in FACE_JOINTS_USE_TAIL:
        return armature_obj.matrix_world @ pose_bone.tail
    return armature_obj.matrix_world @ pose_bone.head


def project_bone_to_camera(scene, camera_obj, armature_obj, pose_bone, width, height, joint_name):
    world_pos = bone_world_head(armature_obj, pose_bone, joint_name)
    co_ndc = world_to_camera_view(scene, camera_obj, world_pos)

    if co_ndc.z < 0:
        return [float("nan"), float("nan")], 0.0, False

    x_px = co_ndc.x * width
    y_px = (1.0 - co_ndc.y) * height

    visible = (0.0 <= co_ndc.x <= 1.0) and (0.0 <= co_ndc.y <= 1.0)
    score = 1.0 if visible else 0.5
    return [float(x_px), float(y_px)], float(score), bool(visible)


def build_symmetry_metadata():
    left = [4, 5, 6, 11, 12, 13]
    right = [1, 2, 3, 14, 15, 16]
    return left, right


_FINGER_SUFFIXES = tuple(
    f"{side}hand{finger}{i}"
    for side in ("left", "right")
    for finger in ("thumb", "index", "middle", "ring", "pinky")
    for i in (1, 2, 3, 4)
)


def _is_finger_bone(norm_name):
    return any(norm_name.endswith(s) for s in _FINGER_SUFFIXES)


def find_source_bone_for_target(target_bone, source_lookup):
    """Returns (pose_bone, rotation_only) or (None, False)."""
    norm_target = normalize_name(target_bone.name)
    is_finger = _is_finger_bone(norm_target)

    # 1) exact normalized name
    src_bone = source_lookup.get(norm_target)
    if src_bone is not None:
        return src_bone, is_finger

    # 2) alias-based lookup
    aliases = RETARGET_BONE_MAP.get(norm_target, [])
    for alias in aliases:
        src_bone = source_lookup.get(normalize_name(alias))
        if src_bone is not None:
            return src_bone, is_finger

    # 3) Map non-thumb fingers to index finger if source lacks them.
    # Use COPY_ROTATION only — these bones sit at different rest positions so
    # copying the full transform would pull them all to the index finger location.
    # Use endswith matching to handle rig name prefixes (e.g. "mixamorig:").
    for finger in ["middle", "ring", "pinky"]:
        for i in (1, 2, 3):
            if norm_target.endswith(f"lefthand{finger}{i}"):
                src = next((v for k, v in source_lookup.items() if k.endswith(f"lefthandindex{i}")), None)
                if src is not None:
                    return src, True
            if norm_target.endswith(f"righthand{finger}{i}"):
                src = next((v for k, v in source_lookup.items() if k.endswith(f"righthandindex{i}")), None)
                if src is not None:
                    return src, True

    # 4) fallback: target distal/tip bones map to previous segment if source has fewer segments
    finger_fallbacks = [
        ("lefthandthumb4", "lefthandthumb3"),
        ("righthandthumb4", "righthandthumb3"),
        ("lefthandindex4", "lefthandindex3"),
        ("righthandindex4", "righthandindex3"),
        ("lefthandmiddle4", "lefthandmiddle3"),
        ("righthandmiddle4", "righthandmiddle3"),
        ("lefthandring4", "lefthandring3"),
        ("righthandring4", "righthandring3"),
        ("lefthandpinky4", "lefthandpinky3"),
        ("righthandpinky4", "righthandpinky3"),
    ]
    fallback_map = dict(finger_fallbacks)

    if norm_target in fallback_map:
        for alias in RETARGET_BONE_MAP.get(fallback_map[norm_target], []):
            src_bone = source_lookup.get(normalize_name(alias))
            if src_bone is not None:
                return src_bone, False

    return None, False


def should_skip_bone_for_retarget(pose_bone):
    name = normalize_name(pose_bone.name)

    helper_tokens = [
        "ik", "pole", "target", "ctrl", "control", "widget",
        "twist", "roll", "end", "socket", "nub", "helper"
    ]
    if any(tok in name for tok in helper_tokens):
        return True

    fingertip_suffixes = (
        "lefthandthumb4", "righthandthumb4",
        "lefthandindex4", "righthandindex4",
        "lefthandmiddle4", "righthandmiddle4",
        "lefthandring4", "righthandring4",
        "lefthandpinky4", "righthandpinky4",
    )
    return name in fingertip_suffixes


def get_camera_intrinsics(camera_obj, width, height):
    """Returns (fx, fy, cx, cy) in pixels from the Blender camera."""
    cam = camera_obj.data
    # Blender stores focal length in mm and sensor size in mm
    if cam.sensor_fit == 'VERTICAL':
        focal_px = (cam.lens / cam.sensor_height) * height
    else:
        # HORIZONTAL or AUTO — use sensor_width against image width
        focal_px = (cam.lens / cam.sensor_width) * width
    cx = width / 2.0
    cy = height / 2.0
    return focal_px, focal_px, cx, cy


def export_videopose3d_style_skeleton(camera_obj, output_dir, armature_obj, frame_start, frame_end, canonical_bones):
    import os
    os.makedirs(output_dir, exist_ok=True)
    npz_path = os.path.join(output_dir, "skeleton_videopose3d.npz")
    if os.path.exists(npz_path):
        print(f"[SKIP] {npz_path} already exists, skipping export_videopose3d_style_skeleton.")
        return

def export_coco_format(camera_obj, output_dir, armature_obj, frame_start, frame_end, canonical_bones):
    """
    Export skeleton in DWPose/COCO order (body + hands) for MusePose finetuning.
    """
    import numpy as np
    import os
    os.makedirs(output_dir, exist_ok=True)
    npz_path = os.path.join(output_dir, "skeleton_coco.npz")
    if os.path.exists(npz_path):
        print(f"[SKIP] {npz_path} already exists, skipping export_coco_format.")
        return
    width, height = get_render_dimensions(scene)
    fx, fy, cx, cy = get_camera_intrinsics(camera_obj, width, height)

    # Compose full COCO/DWPose joint order: body + left hand + right hand
    joint_order = (
        COCO_BODY_JOINTS +
        [j for j in COCO_HAND_LEFT_JOINTS] +
        [j for j in COCO_HAND_RIGHT_JOINTS]
    )

    keypoints_2d = []
    keypoints_3d = []
    scores_2d = []
    visibility_2d = []
    frame_numbers = []

    for frame in range(frame_start, frame_end + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()

        frame_kps_2d = []
        frame_kps_3d = []
        frame_scores = []
        frame_vis = []

        for joint_name in joint_order:
            if joint_name is not None and joint_name in canonical_bones:
                pb = canonical_bones[joint_name]
                xy, score, visible = project_bone_to_camera(
                    scene, camera_obj, armature_obj, pb, width, height, joint_name
                )
                world_pos = bone_world_head(armature_obj, pb, joint_name)
                xyz = [float(world_pos.x), float(world_pos.y), float(world_pos.z)]
            else:
                xy = [float('nan'), float('nan')]
                xyz = [float('nan'), float('nan'), float('nan')]
                score = 0.0
                visible = False
            frame_kps_2d.append(xy)
            frame_kps_3d.append(xyz)
            frame_scores.append(score)
            frame_vis.append(visible)

        keypoints_2d.append(frame_kps_2d)
        keypoints_3d.append(frame_kps_3d)
        scores_2d.append(frame_scores)
        visibility_2d.append(frame_vis)
        frame_numbers.append(frame)

    keypoints_2d = np.asarray(keypoints_2d, dtype=np.float32)
    keypoints_3d = np.asarray(keypoints_3d, dtype=np.float32)
    scores_2d = np.asarray(scores_2d, dtype=np.float32)
    visibility_2d = np.asarray(visibility_2d, dtype=np.bool_)
    frame_numbers = np.asarray(frame_numbers, dtype=np.int32)

    metadata = {
        "layout_name": "coco_wholebody",
        "num_joints": len(joint_order),
        "joint_names": joint_order,
        "video_metadata": {
            camera_obj.name: {
                "w": width,
                "h": height,
                "fps": scene.render.fps,
                "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
            }
        },
        "source_scene": bpy.data.filepath,
        "model_file": MODEL_FILE,
        "animation_file": ANIM_FILE,
    }

    positions_2d = {"subject": {"action": [keypoints_2d]}}
    positions_3d = {"subject": {"action": [keypoints_3d]}}

    npz_path = os.path.join(output_dir, "skeleton_coco.npz")
    np.savez_compressed(
        npz_path,
        positions_2d=np.array(positions_2d, dtype=object),
        positions_3d=np.array(positions_3d, dtype=object),
        metadata=np.array(metadata, dtype=object),
        scores_2d=scores_2d,
        visibility_2d=visibility_2d,
        frame_numbers=frame_numbers,
    )

    json_path = os.path.join(output_dir, "skeleton_coco.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": metadata,
                "frame_numbers": frame_numbers.tolist(),
                "positions_2d_shape": list(keypoints_2d.shape),
                "positions_3d_shape": list(keypoints_3d.shape),
                "scores_2d_shape": list(scores_2d.shape),
            },
            f,
            indent=2,
        )

    print(f"Saved COCO/DWPose-style skeleton data -> {npz_path}")

    width, height = get_render_dimensions(scene)
    fx, fy, cx, cy = get_camera_intrinsics(camera_obj, width, height)

    keypoints_2d = []
    keypoints_3d = []
    scores_2d = []
    visibility_2d = []
    frame_numbers = []

    for frame in range(frame_start, frame_end + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()

        frame_kps_2d = []
        frame_kps_3d = []
        frame_scores = []
        frame_vis = []

        for joint_name in CANONICAL_JOINTS:
            if joint_name in canonical_bones:
                pb = canonical_bones[joint_name]
                xy, score, visible = project_bone_to_camera(
                    scene, camera_obj, armature_obj, pb, width, height, joint_name
                )
                world_pos = bone_world_head(armature_obj, pb, joint_name)
                xyz = [float(world_pos.x), float(world_pos.y), float(world_pos.z)]
            else:
                xy = [float('nan'), float('nan')]
                xyz = [float('nan'), float('nan'), float('nan')]
                score = 0.0
                visible = False
            frame_kps_2d.append(xy)
            frame_kps_3d.append(xyz)
            frame_scores.append(score)
            frame_vis.append(visible)

        keypoints_2d.append(frame_kps_2d)
        keypoints_3d.append(frame_kps_3d)
        scores_2d.append(frame_scores)
        visibility_2d.append(frame_vis)
        frame_numbers.append(frame)

    keypoints_2d = np.asarray(keypoints_2d, dtype=np.float32)
    keypoints_3d = np.asarray(keypoints_3d, dtype=np.float32)
    scores_2d = np.asarray(scores_2d, dtype=np.float32)
    visibility_2d = np.asarray(visibility_2d, dtype=np.bool_)
    frame_numbers = np.asarray(frame_numbers, dtype=np.int32)

    left_joints, right_joints = build_symmetry_metadata()

    metadata = {
        "layout_name": "h36m_17_custom_alias",
        "num_joints": len(CANONICAL_JOINTS),
        "keypoints_symmetry": [left_joints, right_joints],
        "joint_names": CANONICAL_JOINTS,
        "bone_name_map": {k: canonical_bones[k].name for k in CANONICAL_JOINTS if k in canonical_bones},
        "video_metadata": {
            camera_obj.name: {
                "w": width,
                "h": height,
                "fps": scene.render.fps,
                "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
            }
        },
        "source_scene": bpy.data.filepath,
        "model_file": MODEL_FILE,
        "animation_file": ANIM_FILE,
    }

    positions_2d = {"subject": {"action": [keypoints_2d]}}
    positions_3d = {"subject": {"action": [keypoints_3d]}}

    npz_path = os.path.join(output_dir, "skeleton_videopose3d.npz")
    np.savez_compressed(
        npz_path,
        positions_2d=np.array(positions_2d, dtype=object),
        positions_3d=np.array(positions_3d, dtype=object),
        metadata=np.array(metadata, dtype=object),
        scores_2d=scores_2d,
        visibility_2d=visibility_2d,
        frame_numbers=frame_numbers,
    )

    json_path = os.path.join(output_dir, "skeleton_videopose3d.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": metadata,
                "frame_numbers": frame_numbers.tolist(),
                "positions_2d_shape": list(keypoints_2d.shape),
                "positions_3d_shape": list(keypoints_3d.shape),
                "scores_2d_shape": list(scores_2d.shape),
            },
            f,
            indent=2,
        )

    print(f"Saved VideoPose3D-style skeleton data -> {npz_path}")


# --- Call both export functions in your main pipeline ---
# Example usage (add to wherever export_videopose3d_style_skeleton is called):
# export_coco_format(camera_obj, output_dir, armature_obj, frame_start, frame_end, canonical_bones)


def load_rendered_frame(frame_path):
    if Image is None:
        raise RuntimeError("Pillow is required for overlay rendering. Install pillow into Blender's Python.")
    return Image.open(frame_path).convert("RGB")


def draw_joint(draw, x, y, radius=4, color=(255, 255, 255)):
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill=color,
        outline=(0, 0, 0),
        width=1,
    )


def draw_bone(draw, p1, p2, color=(255, 255, 255), width=4):
    draw.line((p1[0], p1[1], p2[0], p2[1]), fill=color, width=width)


def finite_point(p):
    return (
        p is not None
        and len(p) == 2
        and not math.isnan(p[0])
        and not math.isnan(p[1])
    )


def render_overlay_frames(camera_obj, output_dir, armature_obj, frame_start, frame_end, canonical_bones):
    os.makedirs(output_dir, exist_ok=True)
    width, height = get_render_dimensions(scene)

    joint_index = {name: i for i, name in enumerate(CANONICAL_JOINTS)}

    for frame in range(frame_start, frame_end + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()

        frame_kps = {}
        # Calculate all joints except pelvis first
        for joint_name in CANONICAL_JOINTS:
            if joint_name == "pelvis":
                continue
            if joint_name in canonical_bones:
                pb = canonical_bones[joint_name]
                xy, score, visible = project_bone_to_camera(
                    scene, camera_obj, armature_obj, pb, width, height, joint_name
                )
            else:
                xy = [float('nan'), float('nan')]
                score = 0.0
                visible = False
            frame_kps[joint_name] = {
                "xy": xy,
                "score": score,
                "visible": visible,
            }

        # Calculate pelvis as midpoint between left_hip and right_hip, only if both hips are present
        if "left_hip" in frame_kps and "right_hip" in frame_kps:
            left_hip_xy = frame_kps["left_hip"]["xy"]
            right_hip_xy = frame_kps["right_hip"]["xy"]
            pelvis_xy = [
                0.5 * (left_hip_xy[0] + right_hip_xy[0]),
                0.5 * (left_hip_xy[1] + right_hip_xy[1])
            ]
            pelvis_visible = (
                frame_kps["left_hip"]["visible"] and frame_kps["right_hip"]["visible"]
            )
            pelvis_score = 1.0 if pelvis_visible else 0.5
            frame_kps["pelvis"] = {
                "xy": pelvis_xy,
                "score": pelvis_score,
                "visible": pelvis_visible,
            }

        raw_path = os.path.join(output_dir, f"frame_{frame:04d}.png")
        overlay_path = os.path.join(output_dir, f"overlay_{frame:04d}.png")

        if not os.path.exists(raw_path):
            print(f"Skipping overlay for missing frame: {raw_path}")
            continue

        img = load_rendered_frame(raw_path)
        draw = ImageDraw.Draw(img)

        # bones first
        for edge in SKELETON_EDGES:
            a, b = edge
            if a in frame_kps and b in frame_kps:
                pa = frame_kps[a]["xy"]
                pb = frame_kps[b]["xy"]

                if finite_point(pa) and finite_point(pb):
                    draw_bone(draw, pa, pb, color=EDGE_COLORS.get(edge, (255, 255, 255)), width=5)

        # joints second
        for joint_name in CANONICAL_JOINTS:
            if joint_name in frame_kps:
                p = frame_kps[joint_name]["xy"]
                if finite_point(p):
                    draw_joint(draw, p[0], p[1], radius=5, color=(255, 255, 255))

        img.save(overlay_path)
        print(f"Saved overlay frame -> {overlay_path}")


def _sample_joints(joint_list, canonical_bones, armature_obj, camera_obj, width, height):
    """
    For a single frame (caller must set scene.frame_set beforehand), sample 2D pixel
    coords, world-space 3D, camera-space 3D (OpenCV convention), and visibility for
    every joint in joint_list. Returns (kps_2d, kps_3d_world, kps_3d_cam, scores, vis).
    Each list entry corresponds to one joint; missing joints get NaN.
    """
    cam_inv = camera_obj.matrix_world.inverted()
    nan2 = [float("nan"), float("nan")]
    nan3 = [float("nan"), float("nan"), float("nan")]

    kps_2d, kps_3d_world, kps_3d_cam, scores, vis = [], [], [], [], []
    for joint_name in joint_list:
        if joint_name is not None and joint_name in canonical_bones:
            pb = canonical_bones[joint_name]
            xy, score, visible = project_bone_to_camera(
                scene, camera_obj, armature_obj, pb, width, height, joint_name
            )
            wp = bone_world_head(armature_obj, pb, joint_name)
            cp = cam_inv @ wp
            # Blender camera: X right, Y up, -Z forward → OpenCV: X right, Y down, Z forward
            xyz_cam = [float(cp.x), float(-cp.y), float(-cp.z)]
            xyz_world = [float(wp.x), float(wp.y), float(wp.z)]
        else:
            xy, score, visible = nan2, 0.0, False
            xyz_world, xyz_cam = nan3, nan3
        kps_2d.append(xy)
        kps_3d_world.append(xyz_world)
        kps_3d_cam.append(xyz_cam)
        scores.append(score)
        vis.append(visible)
    return kps_2d, kps_3d_world, kps_3d_cam, scores, vis


def export_h36m_format(camera_obj, output_dir, armature_obj, frame_start, frame_end, canonical_bones):
    """
    Export a VideoPose3D-compatible NPZ with H36M 17-joint layout.
    2D coords are normalized to [0, 1] by image dimensions.
    3D coords are in camera space (OpenCV convention, metres).
    Camera intrinsics and extrinsics are included for lifting.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    npz_path = os.path.join(output_dir, "skeleton_h36m.npz")
    if os.path.exists(npz_path):
        print(f"[SKIP] {npz_path} already exists, skipping export_h36m_format.")
        return
    width, height = get_render_dimensions(scene)
    fx, fy, cx, cy = get_camera_intrinsics(camera_obj, width, height)

    # Camera extrinsics: world → camera rotation (3×3) and translation (3,)
    cam_inv = camera_obj.matrix_world.inverted()
    R = [[cam_inv[r][c] for c in range(3)] for r in range(3)]
    t = [cam_inv[r][3] for r in range(3)]

    kps2_frames, kps3_frames, score_frames, vis_frames = [], [], [], []

    for frame in range(frame_start, frame_end + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        kps_2d, _, kps_3d_cam, scores, vis = _sample_joints(
            H36M_JOINTS, canonical_bones, armature_obj, camera_obj, width, height
        )
        # Normalize 2D to [0, 1]
        kps_2d_norm = [
            [xy[0] / width, xy[1] / height] for xy in kps_2d
        ]
        kps2_frames.append(kps_2d_norm)
        kps3_frames.append(kps_3d_cam)
        score_frames.append(scores)
        vis_frames.append(vis)

    kps2 = np.asarray(kps2_frames, dtype=np.float32)   # (F, 17, 2)
    kps3 = np.asarray(kps3_frames, dtype=np.float32)   # (F, 17, 3)
    scores = np.asarray(score_frames, dtype=np.float32)
    vis = np.asarray(vis_frames, dtype=np.bool_)

    metadata = {
        "layout": "h36m_17",
        "joint_names": H36M_JOINTS,
        "keypoints_symmetry": [H36M_LEFT_JOINTS, H36M_RIGHT_JOINTS],
        "image_size": [width, height],
        "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        "extrinsics": {"R": R, "t": t},
        "coord_system_3d": "camera_opencv",  # X right, Y down, Z forward
        "coord_units": "blender_metres",
        "model_file": MODEL_FILE,
        "animation_file": ANIM_FILE,
    }

    npz_path = os.path.join(output_dir, "skeleton_h36m.npz")
    np.savez_compressed(
        npz_path,
        positions_2d=kps2,
        positions_3d=kps3,
        scores=scores,
        visibility=vis,
        metadata=np.array(metadata, dtype=object),
    )
    print(f"Saved H36M skeleton -> {npz_path}")


def export_coco_format(camera_obj, output_dir, armature_obj, frame_start, frame_end, canonical_bones):
    """
    Export a DWPose-compatible NPZ with COCO body (17) + left hand (21) + right hand (21).
    2D coords are in pixels. 3D coords are in camera space (OpenCV convention).
    Matches the layout DWPose wholebody produces so arrays can be compared directly.
    """
    width, height = get_render_dimensions(scene)
    fx, fy, cx, cy = get_camera_intrinsics(camera_obj, width, height)

    body2_frames, body3_frames, body_scores, body_vis = [], [], [], []
    lh2_frames,   lh3_frames,   lh_scores,   lh_vis   = [], [], [], []
    rh2_frames,   rh3_frames,   rh_scores,   rh_vis   = [], [], [], []

    for frame in range(frame_start, frame_end + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()

        b2, _, b3, bs, bv = _sample_joints(
            COCO_BODY_JOINTS, canonical_bones, armature_obj, camera_obj, width, height
        )
        l2, _, l3, ls, lv = _sample_joints(
            COCO_HAND_LEFT_JOINTS, canonical_bones, armature_obj, camera_obj, width, height
        )
        r2, _, r3, rs, rv = _sample_joints(
            COCO_HAND_RIGHT_JOINTS, canonical_bones, armature_obj, camera_obj, width, height
        )
        body2_frames.append(b2);  body3_frames.append(b3)
        body_scores.append(bs);   body_vis.append(bv)
        lh2_frames.append(l2);    lh3_frames.append(l3)
        lh_scores.append(ls);     lh_vis.append(lv)
        rh2_frames.append(r2);    rh3_frames.append(r3)
        rh_scores.append(rs);     rh_vis.append(rv)

    def _arr(lst, dtype): return np.asarray(lst, dtype=dtype)

    metadata = {
        "layout": "coco_body17_hand21x2",
        "body_joint_names": COCO_BODY_JOINTS,
        "left_hand_joint_names": COCO_HAND_LEFT_JOINTS,
        "right_hand_joint_names": COCO_HAND_RIGHT_JOINTS,
        "image_size": [width, height],
        "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        "coord_system_3d": "camera_opencv",
        "coord_units": "blender_metres",
        "model_file": MODEL_FILE,
        "animation_file": ANIM_FILE,
    }

    npz_path = os.path.join(output_dir, "skeleton_coco.npz")
    np.savez_compressed(
        npz_path,
        body_kps_2d=_arr(body2_frames, np.float32),    # (F, 17, 2)
        body_kps_3d=_arr(body3_frames, np.float32),    # (F, 17, 3)
        body_scores=_arr(body_scores,  np.float32),
        body_vis=_arr(body_vis,        np.bool_),
        left_hand_kps_2d=_arr(lh2_frames, np.float32),  # (F, 21, 2)
        left_hand_kps_3d=_arr(lh3_frames, np.float32),
        left_hand_scores=_arr(lh_scores,  np.float32),
        left_hand_vis=_arr(lh_vis,        np.bool_),
        right_hand_kps_2d=_arr(rh2_frames, np.float32), # (F, 21, 2)
        right_hand_kps_3d=_arr(rh3_frames, np.float32),
        right_hand_scores=_arr(rh_scores,  np.float32),
        right_hand_vis=_arr(rh_vis,        np.bool_),
        metadata=np.array(metadata, dtype=object),
    )
    print(f"Saved COCO skeleton -> {npz_path}")


def key_camera_follow_root(camera_obj, armature_obj, frame_start, frame_end, canonical_bones):
    """
    Keys the camera XZ translation each frame so it tracks the model's pelvis
    movement in world space. Depth (Y) and rotation are left unchanged, keeping
    the same viewing angle and zoom throughout.
    """
    root_pb = canonical_bones.get("pelvis")
    if root_pb is None:
        print("Warning: no pelvis bone found, camera will not track model movement")
        return

    scene.frame_set(frame_start)
    bpy.context.view_layer.update()
    ref_pos = (armature_obj.matrix_world @ root_pb.head).copy()
    cam_start = camera_obj.location.copy()

    for frame in range(frame_start, frame_end + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        cur_pos = armature_obj.matrix_world @ root_pb.head
        camera_obj.location.x = cam_start.x + (cur_pos.x - ref_pos.x)
        # Z is intentionally NOT tracked — camera stays at its starting height
        # so it doesn't drop when the character falls or goes to the ground.
        camera_obj.keyframe_insert(data_path="location", frame=frame)

    scene.frame_set(frame_start)
    bpy.context.view_layer.update()
    print(f"Keyed camera '{camera_obj.name}' to follow model root over {frame_end - frame_start + 1} frames")


def _hsv1_to_rgb255(h):
    """h in [0,1], s=v=1 → (r,g,b) each in [0,255] — matches matplotlib.colors.hsv_to_rgb * 255."""
    h6 = h * 6.0
    i = int(h6) % 6
    f = h6 - int(h6)
    if   i == 0: r, g, b = 1.0,   f,   0.0
    elif i == 1: r, g, b = 1.0-f, 1.0, 0.0
    elif i == 2: r, g, b = 0.0,   1.0, f
    elif i == 3: r, g, b = 0.0,   1.0-f, 1.0
    elif i == 4: r, g, b = f,     0.0, 1.0
    else:        r, g, b = 1.0,   0.0, 1.0-f
    return (int(r * 255), int(g * 255), int(b * 255))  # RGB order (matches matplotlib output)


def _dwpose_smart_width(d):
    if d < 5:   return 1
    if d < 10:  return 2
    if d < 20:  return 3
    if d < 40:  return 4
    if d < 80:  return 5
    if d < 160: return 6
    if d < 320: return 7
    return 8


def _ellipse2poly(center, axes, angle_deg):
    """
    Pure-Python replacement for cv2.ellipse2Poly.
    Returns a list of (x, y) integer tuples tracing the ellipse perimeter.
    """
    cx, cy = center
    a, b = max(1, axes[0]), max(1, axes[1])
    cos_a = math.cos(math.radians(angle_deg))
    sin_a = math.sin(math.radians(angle_deg))
    points = []
    for t_deg in range(0, 360):
        t = math.radians(t_deg)
        x = a * math.cos(t)
        y = b * math.sin(t)
        points.append((int(cos_a * x - sin_a * y + cx),
                       int(sin_a * x + cos_a * y + cy)))
    return points


def _dwpose_draw_body(canvas, candidate, subset, limb_colors=None):
    """
    PIL port of MusePose util.draw_bodypose (single actor, red/dark-red scheme).
    candidate   : (18, 2) float32, normalized [0,1] (x, y)
    subset      : (1, 18) float32, index into candidate or -1 if joint absent
    limb_colors : optional list of 18 RGB tuples, one per limb in limb_seq order.
                  Defaults to MODEL_A_BODY_LIMB_COLORS.
    """
    H, W = canvas.shape[:2]
    limb_seq = [
        [2,3],[2,6],[3,4],[4,5],[6,7],[7,8],[2,9],[9,10],
        [10,11],[2,12],[12,13],[13,14],[2,1],[1,15],[15,17],
        [1,16],[16,18],
    ]
    colors = limb_colors if limb_colors is not None else MODEL_A_BODY_LIMB_COLORS

    # Draw limb ellipses onto PIL image
    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)

    for i in range(len(limb_seq)):
        color = colors[i]
        for n in range(len(subset)):
            idx = subset[n][np.array(limb_seq[i]) - 1]
            if -1 in idx:
                continue
            Y = candidate[idx.astype(int), 0] * float(W)
            X = candidate[idx.astype(int), 1] * float(H)
            mX, mY = np.mean(X), np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            w = _dwpose_smart_width(length)
            poly = _ellipse2poly((int(mY), int(mX)), (int(length / 2), w), angle)
            if len(poly) >= 3:
                draw.polygon(poly, fill=color)

    # Darken (matches original canvas * 0.6)
    canvas = (np.array(img) * 0.6).astype(np.uint8)

    # Draw joint circles: correct per-side color + 1px white outline
    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    for i in range(18):
        limb_idx = JOINT_TO_LIMB_COLOR[i] if i < len(JOINT_TO_LIMB_COLOR) else 12
        color = colors[limb_idx] if limb_idx < len(colors) else colors[12]
        for n in range(len(subset)):
            idx = int(subset[n][i])
            if idx == -1:
                continue
            x, y = candidate[idx][0:2]
            px, py, r = int(x * W), int(y * H), 4
            draw.ellipse([px - r - 1, py - r - 1, px + r + 1, py + r + 1], fill=(255, 255, 255))
            draw.ellipse([px - r, py - r, px + r, py + r], fill=color)

    return np.array(img)


def _dwpose_draw_hands(canvas, all_hand_peaks, hand_colors=None):
    """
    PIL port of MusePose util.draw_handpose.
    all_hand_peaks : (2, 21, 2) float32, normalized [0,1] (x, y); 0=left, 1=right hand.
    hand_colors    : optional list of 2 base RGB tuples [left_color, right_color].
                     Edges are shaded dark (knuckle) → light (fingertip).
                     Defaults to MODEL_A_HAND_COLORS.
    """
    H, W = canvas.shape[:2]
    edges = [
        [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],
        [10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20],
    ]
    hcolors = hand_colors if hand_colors is not None else MODEL_A_HAND_COLORS
    eps = 0.01
    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    for hand_idx, peaks in enumerate(all_hand_peaks):
        base = hcolors[hand_idx % len(hcolors)]
        peaks = np.array(peaks)
        for ie, e in enumerate(edges):
            x1, y1 = peaks[e[0]]
            x2, y2 = peaks[e[1]]
            x1, y1, x2, y2 = int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)
            if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                w = _dwpose_smart_width(((x1-x2)**2 + (y1-y2)**2)**0.5)
                # t = 0 at knuckle (dark), t = 1 at fingertip (light)
                t = (ie % 4) / 3.0
                r = int(min(255, base[0] + int(t * (255 - base[0]))))
                g = int(min(255, base[1] + int(t * (255 - base[1]))))
                b = int(min(255, base[2] + int(t * (255 - base[2]))))
                color = (r, g, b)
                draw.line([(x1, y1), (x2, y2)], fill=color, width=w)
        for kp in peaks:
            x, y = int(kp[0] * W), int(kp[1] * H)
            if x > eps and y > eps:
                r = 3
                draw.ellipse([x - r - 1, y - r - 1, x + r + 1, y + r + 1], fill=(255, 255, 255))
                draw.ellipse([x - r, y - r, x + r, y + r], fill=base)
    return np.array(img)


def render_dwpose_frames(camera_obj, output_dir, armature_obj, frame_start, frame_end, canonical_bones):
    """
    Renders per-frame skeleton images in MusePose's DWPose visual format using
    ground-truth 3D bone positions projected through the camera.

    Produces dwpose_%04d.png per frame.  These frames are compiled into
    {cam_name}_dwpose.mp4 by dataset_pipeline.py, which is used directly as
    kps_path in the MusePose training meta JSON — giving exact ground-truth
    pose signal without any detector error.
    """
    if not PIL_AVAILABLE:
        print("Skipping DWPose frame rendering: Pillow not available")
        return

    os.makedirs(output_dir, exist_ok=True)
    width, height = get_render_dimensions(scene)

    for frame in range(frame_start, frame_end + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()

        # ── Body: 18 OpenPose joints ──────────────────────────────────────────
        candidate = np.zeros((18, 2), dtype=np.float32)
        subset    = np.full((1, 18), -1.0, dtype=np.float32)

        for j, joint_name in enumerate(DWPOSE_BODY18_JOINTS):
            # Neck (index 1) is computed below as shoulder midpoint
            if j == 1:
                continue
            if joint_name is None or joint_name not in canonical_bones:
                continue
            pb = canonical_bones[joint_name]
            xy, _, visible = project_bone_to_camera(
                scene, camera_obj, armature_obj, pb, width, height, joint_name
            )
            if visible and not (math.isnan(xy[0]) or math.isnan(xy[1])):
                candidate[j] = [xy[0] / width, xy[1] / height]
                subset[0][j] = float(j)

        # Neck: midpoint of left_shoulder (5) and right_shoulder (2)
        ls_idx, rs_idx = 5, 2
        if subset[0][ls_idx] != -1 and subset[0][rs_idx] != -1:
            candidate[1] = (candidate[ls_idx] + candidate[rs_idx]) * 0.5
            subset[0][1] = 1.0

        # ── Hands: 21 joints each ────────────────────────────────────────────
        def collect_hand(joint_list):
            hand = np.zeros((21, 2), dtype=np.float32)
            for k, jname in enumerate(joint_list):
                if jname is None or jname not in canonical_bones:
                    continue
                pb = canonical_bones[jname]
                xy, _, visible = project_bone_to_camera(
                    scene, camera_obj, armature_obj, pb, width, height, jname
                )
                if visible and not (math.isnan(xy[0]) or math.isnan(xy[1])):
                    hand[k] = [xy[0] / width, xy[1] / height]
            return hand

        left_hand  = collect_hand(COCO_HAND_LEFT_JOINTS)
        right_hand = collect_hand(COCO_HAND_RIGHT_JOINTS)
        all_hands  = np.stack([left_hand, right_hand], axis=0)

        # ── Draw and save — custom palette ────────────────────────────────────
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas = _dwpose_draw_body(canvas, candidate, subset, limb_colors=MODEL_A_BODY_LIMB_COLORS)
        canvas = _dwpose_draw_hands(canvas, all_hands, hand_colors=MODEL_A_HAND_COLORS)
        Image.fromarray(canvas).save(os.path.join(output_dir, f"dwpose_{frame:04d}.png"))

        # ── Draw and save — standard MusePose rainbow palette ─────────────────
        canvas_rb = np.zeros((height, width, 3), dtype=np.uint8)
        canvas_rb = _dwpose_draw_body(canvas_rb, candidate, subset, limb_colors=RAINBOW_BODY_LIMB_COLORS)
        canvas_rb = _dwpose_draw_hands(canvas_rb, all_hands, hand_colors=RAINBOW_HAND_COLORS)
        Image.fromarray(canvas_rb).save(os.path.join(output_dir, f"dwpose_rainbow_{frame:04d}.png"))

    print(f"DWPose frames saved -> {output_dir}")


def render_dwpose_multiactor_frames(camera_obj, output_dir, actors_data, frame_start, frame_end):
    """
    Renders per-frame DWPose skeleton images with all actors drawn on a single canvas.
    Each actor is drawn with a distinct color from ACTOR_BODY_COLORS.
    Saves dwpose_multiactor_%04d.png files used as the kps_path for multi-actor MusePose training.
    """
    if not PIL_AVAILABLE:
        print("Skipping multiactor DWPose rendering: Pillow not available")
        return

    os.makedirs(output_dir, exist_ok=True)
    width, height = get_render_dimensions(scene)

    for frame in range(frame_start, frame_end + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()

        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        for actor_idx, (_, _, armature_obj, canonical_bones) in enumerate(actors_data):
            limb_colors = ACTOR_BODY_LIMB_COLORS[actor_idx % len(ACTOR_BODY_LIMB_COLORS)]
            hnd_colors  = ACTOR_HAND_COLORS[actor_idx % len(ACTOR_HAND_COLORS)]

            # Body: 18 OpenPose joints
            candidate = np.zeros((18, 2), dtype=np.float32)
            subset    = np.full((1, 18), -1.0, dtype=np.float32)

            for j, joint_name in enumerate(DWPOSE_BODY18_JOINTS):
                if j == 1:
                    continue
                if joint_name is None or joint_name not in canonical_bones:
                    continue
                pb = canonical_bones[joint_name]
                xy, _, visible = project_bone_to_camera(
                    scene, camera_obj, armature_obj, pb, width, height, joint_name
                )
                if visible and not (math.isnan(xy[0]) or math.isnan(xy[1])):
                    candidate[j] = [xy[0] / width, xy[1] / height]
                    subset[0][j] = float(j)

            ls_idx, rs_idx = 5, 2
            if subset[0][ls_idx] != -1 and subset[0][rs_idx] != -1:
                candidate[1] = (candidate[ls_idx] + candidate[rs_idx]) * 0.5
                subset[0][1] = 1.0

            canvas = _dwpose_draw_body(canvas, candidate, subset, limb_colors=limb_colors)

            def _collect_hand(joint_list, arm_obj, can_bones):
                hand = np.zeros((21, 2), dtype=np.float32)
                for k, jname in enumerate(joint_list):
                    if jname is None or jname not in can_bones:
                        continue
                    pb = can_bones[jname]
                    xy, _, visible = project_bone_to_camera(
                        scene, camera_obj, arm_obj, pb, width, height, jname
                    )
                    if visible and not (math.isnan(xy[0]) or math.isnan(xy[1])):
                        hand[k] = [xy[0] / width, xy[1] / height]
                return hand

            left_hand  = _collect_hand(COCO_HAND_LEFT_JOINTS,  armature_obj, canonical_bones)
            right_hand = _collect_hand(COCO_HAND_RIGHT_JOINTS, armature_obj, canonical_bones)
            canvas = _dwpose_draw_hands(canvas, np.stack([left_hand, right_hand], axis=0), hand_colors=hnd_colors)

        out_path = os.path.join(output_dir, f"dwpose_multiactor_{frame:04d}.png")
        Image.fromarray(canvas).save(out_path)

    print(f"Multiactor DWPose frames saved -> {output_dir}")


def export_multiactor_coco_format(camera_obj, output_dir, actors_data, frame_start, frame_end):
    """
    Export combined multi-actor COCO wholebody skeleton data.

    Output arrays have shape (F, N_actors, J, 2or3):
      body_kps_2d  : (F, N, 17, 2)  pixel-space
      body_kps_3d  : (F, N, 17, 3)  camera-space OpenCV
      left_hand_*  : (F, N, 21, 2/3)
      right_hand_* : (F, N, 21, 2/3)

    This format is designed for training a multi-actor MusePose variant where the
    pose signal must encode all actors simultaneously.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    npz_path = os.path.join(output_dir, "skeleton_multiactor_coco.npz")
    if os.path.exists(npz_path):
        print(f"[SKIP] {npz_path} already exists, skipping export_multiactor_coco_format.")
        return
    width, height = get_render_dimensions(scene)
    fx, fy, cx, cy = get_camera_intrinsics(camera_obj, width, height)

    N = len(actors_data)

    all_body2 = [[] for _ in range(N)]
    all_body3 = [[] for _ in range(N)]
    all_lh2   = [[] for _ in range(N)]
    all_lh3   = [[] for _ in range(N)]
    all_rh2   = [[] for _ in range(N)]
    all_rh3   = [[] for _ in range(N)]

    for frame in range(frame_start, frame_end + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()

        for i, (_, _, armature_obj, canonical_bones) in enumerate(actors_data):
            b2, _, b3, _, _ = _sample_joints(
                COCO_BODY_JOINTS, canonical_bones, armature_obj, camera_obj, width, height
            )
            l2, _, l3, _, _ = _sample_joints(
                COCO_HAND_LEFT_JOINTS, canonical_bones, armature_obj, camera_obj, width, height
            )
            r2, _, r3, _, _ = _sample_joints(
                COCO_HAND_RIGHT_JOINTS, canonical_bones, armature_obj, camera_obj, width, height
            )
            all_body2[i].append(b2);  all_body3[i].append(b3)
            all_lh2[i].append(l2);    all_lh3[i].append(l3)
            all_rh2[i].append(r2);    all_rh3[i].append(r3)

    def _arr(lst): return np.asarray(lst, dtype=np.float32)

    # Stack to (F, N, J, 2or3)
    body_kps_2d = np.stack([_arr(all_body2[i]) for i in range(N)], axis=1)
    body_kps_3d = np.stack([_arr(all_body3[i]) for i in range(N)], axis=1)
    lh_kps_2d   = np.stack([_arr(all_lh2[i])   for i in range(N)], axis=1)
    lh_kps_3d   = np.stack([_arr(all_lh3[i])   for i in range(N)], axis=1)
    rh_kps_2d   = np.stack([_arr(all_rh2[i])   for i in range(N)], axis=1)
    rh_kps_3d   = np.stack([_arr(all_rh3[i])   for i in range(N)], axis=1)

    actors_meta = [
        {"actor_idx": i, "model_file": actors_data[i][0], "anim_file": actors_data[i][1]}
        for i in range(N)
    ]

    metadata = {
        "layout": "multiactor_coco_body17_hand21x2",
        "num_actors": N,
        "actors": actors_meta,
        "body_joint_names": COCO_BODY_JOINTS,
        "left_hand_joint_names": COCO_HAND_LEFT_JOINTS,
        "right_hand_joint_names": COCO_HAND_RIGHT_JOINTS,
        "image_size": [width, height],
        "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        "coord_system_3d": "camera_opencv",
        "coord_units": "blender_metres",
        "array_axes": "(F, N_actors, J, 2or3)",
    }

    npz_path = os.path.join(output_dir, "skeleton_multiactor_coco.npz")
    np.savez_compressed(
        npz_path,
        body_kps_2d=body_kps_2d,
        body_kps_3d=body_kps_3d,
        left_hand_kps_2d=lh_kps_2d,
        left_hand_kps_3d=lh_kps_3d,
        right_hand_kps_2d=rh_kps_2d,
        right_hand_kps_3d=rh_kps_3d,
        metadata=np.array(metadata, dtype=object),
    )
    print(f"Saved multiactor COCO skeleton -> {npz_path}  shape: {body_kps_2d.shape}")


def render_camera_sequence(camera_obj, output_dir, actors_data, frame_start, frame_end):
    import time
    os.makedirs(output_dir, exist_ok=True)

    scene.camera = camera_obj
    scene.frame_start = frame_start
    scene.frame_end = frame_end
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = os.path.join(output_dir, "frame_")

    # Camera follows actor 0's root
    _, _, primary_armature, primary_bones = actors_data[0]
    key_camera_follow_root(camera_obj, primary_armature, frame_start, frame_end, primary_bones)

    if len(actors_data) == 1:
        # Single-actor: original output paths (backward compatible)
        _, _, armature_obj, canonical_bones = actors_data[0]
        kwargs = dict(
            camera_obj=camera_obj,
            output_dir=output_dir,
            armature_obj=armature_obj,
            frame_start=frame_start,
            frame_end=frame_end,
            canonical_bones=canonical_bones,
        )
        export_videopose3d_style_skeleton(**kwargs)
        export_h36m_format(**kwargs)
        export_coco_format(**kwargs)
    else:
        # Multi-actor: per-actor skeleton data in actor_N subdirs
        for actor_idx, (_, _, armature_obj, canonical_bones) in enumerate(actors_data):
            actor_dir = os.path.join(output_dir, f"actor_{actor_idx}")
            os.makedirs(actor_dir, exist_ok=True)
            kwargs = dict(
                camera_obj=camera_obj,
                output_dir=actor_dir,
                armature_obj=armature_obj,
                frame_start=frame_start,
                frame_end=frame_end,
                canonical_bones=canonical_bones,
            )
            export_h36m_format(**kwargs)
            export_coco_format(**kwargs)
        # Combined multi-actor skeleton (F, N, J, 2/3)
        export_multiactor_coco_format(camera_obj, output_dir, actors_data, frame_start, frame_end)


    print(f"Rendering camera {camera_obj.name} -> {output_dir}")
    t0 = time.time()
    total_frames = frame_end - frame_start + 1
    for idx, frame in enumerate(range(frame_start, frame_end + 1), 1):
        t1 = time.time()
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        # Set output path for this frame
        bpy.context.scene.render.filepath = os.path.join(output_dir, f"frame_{frame:04d}")
        t2 = time.time()
        print(f"  Rendering frame {idx}/{total_frames} (frame {frame})...")
        bpy.ops.render.render(write_still=True)
        t3 = time.time()
        print(f"    [TIMER] Frame setup: {t2-t1:.2f}s, Frame render: {t3-t2:.2f}s")
    t4 = time.time()
    print(f"  [TIMER] Camera {camera_obj.name} total: {t4-t0:.2f}s")

    if len(actors_data) == 1:
        _, _, armature_obj, canonical_bones = actors_data[0]
        render_dwpose_frames(
            camera_obj=camera_obj, output_dir=output_dir,
            armature_obj=armature_obj, frame_start=frame_start,
            frame_end=frame_end, canonical_bones=canonical_bones,
        )
    else:
        # Per-actor single-color DWPose in actor subdirs
        for actor_idx, (_, _, armature_obj, canonical_bones) in enumerate(actors_data):
            actor_dir = os.path.join(output_dir, f"actor_{actor_idx}")
            render_dwpose_frames(
                camera_obj=camera_obj, output_dir=actor_dir,
                armature_obj=armature_obj, frame_start=frame_start,
                frame_end=frame_end, canonical_bones=canonical_bones,
            )
        # Combined multi-actor DWPose canvas
        render_dwpose_multiactor_frames(camera_obj, output_dir, actors_data, frame_start, frame_end)


def fit_cameras_to_actors(cameras, actors_all_objs, frame_start, width, height, margin=0.98):
    """
    Adjusts each camera's focal length to the maximum zoom level that keeps all actor
    meshes within frame — i.e., zooms in until one model just reaches the frame edge.

    Uses the rest-pose axis-aligned bounding box of each mesh transformed by its
    current matrix_world (valid after baking, since the armature object is positioned
    at the spawn point).

    margin : 0.98 leaves a 2% border so floating-point edge cases don't clip pixels.
    """
    # Evaluate scene at first frame so matrix_world is up to date for all objects.
    bpy.context.scene.frame_set(frame_start)
    bpy.context.view_layer.update()

    # Collect world-space bounding box corners from every actor mesh.
    all_corners = []
    for model_objs in actors_all_objs:
        for obj in model_objs:
            if obj.type != "MESH":
                continue
            for corner in obj.bound_box:
                all_corners.append(obj.matrix_world @ Vector(corner))

    if not all_corners:
        print("fit_cameras_to_actors: no mesh objects found, skipping")
        return

    aspect = height / width  # e.g. 1.0 for 1024×1024

    for cam in cameras:
        cam_inv = cam.matrix_world.inverted()

        # Sensor dimensions in mm, adjusted for image aspect ratio.
        sensor_w = cam.data.sensor_width
        if cam.data.sensor_fit == "VERTICAL":
            sensor_h = sensor_w
            sensor_w = sensor_h / aspect
        else:  # HORIZONTAL or AUTO
            sensor_h = sensor_w * aspect

        max_px_ratio = 1e-9  # max( |px| / depth ) across all corners
        max_py_ratio = 1e-9  # max( |py| / depth ) across all corners
        any_visible  = False

        for wco in all_corners:
            p = cam_inv @ wco
            # Camera looks along -Z; positive depth means in front of camera.
            if p.z >= -0.001:
                continue  # behind or at camera plane, skip
            depth = abs(p.z)
            max_px_ratio = max(max_px_ratio, abs(p.x) / depth)
            max_py_ratio = max(max_py_ratio, abs(p.y) / depth)
            any_visible  = True

        if not any_visible:
            print(f"  {cam.name}: all actor bounds behind camera, skipping")
            continue

        # Maximum focal length that fits each axis — take the more restrictive one.
        fl_h = sensor_w / (2.0 * max_px_ratio)
        fl_v = sensor_h / (2.0 * max_py_ratio)
        optimal_fl = min(fl_h, fl_v) * margin

        # Clamp to a sane range (10 mm fish-eye → 600 mm telephoto).
        optimal_fl = max(10.0, min(600.0, optimal_fl))

        old_fl = cam.data.lens
        cam.data.lens = optimal_fl
        print(f"  {cam.name}: focal length {old_fl:.1f} mm -> {optimal_fl:.1f} mm")


def _get_action_fcurves(action):
    """
    Return all FCurves from a Blender Action, compatible with Blender 4.x and 5.0+.

    Blender 4.x:  action.fcurves (direct attribute)
    Blender 5.0+: action.layers[i].strips[j].channelbags[k].fcurves  (slotted/layered API)
                  fallback: action.layers[i].strips[j].keyframe_data.fcurves
    """
    # Legacy path (Blender < 5.0, or legacy-format actions in 5.0)
    if getattr(action, "is_action_legacy", False):
        return list(action.fcurves)
    if hasattr(action, "fcurves") and not hasattr(action, "layers"):
        return list(action.fcurves)

    # Layered action path (Blender 5.0+)
    result = []
    for layer in getattr(action, "layers", []):
        for strip in getattr(layer, "strips", []):
            # Primary 5.0 API: channelbags
            if hasattr(strip, "channelbags"):
                for cb in strip.channelbags:
                    result.extend(getattr(cb, "fcurves", []))
            # Intermediate/nightly build fallback
            elif hasattr(strip, "keyframe_data") and hasattr(strip.keyframe_data, "fcurves"):
                result.extend(strip.keyframe_data.fcurves)
    return result


def apply_floor_snap(target_armature, canonical_bones, model_objs, floor_z, frame_start, frame_end):
    """
    After baking, shift target_armature's Z location keyframes so the lowest
    mesh vertex across the ENTIRE animation lands exactly on floor_z.

    Scans every frame so the correction accounts for the character's lowest
    point throughout the clip — not just the starting pose.  Uses the evaluated
    (deformed) mesh, so it works regardless of pose (standing, handstand, etc.).
    Falls back to bone endpoints if no mesh objects are present.
    """
    mesh_objs = [o for o in model_objs if o.type == "MESH"]

    global_min_z = float("inf")
    frame_count = frame_end - frame_start + 1
    print(f"  Floor snap: scanning {frame_count} frames for global mesh minimum...")

    for frame in range(frame_start, frame_end + 1):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()

        if mesh_objs:
            for obj in mesh_objs:
                eval_obj = obj.evaluated_get(depsgraph)
                mesh = eval_obj.to_mesh()
                # Fast bulk vertex extraction via numpy
                verts = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
                mesh.vertices.foreach_get("co", verts)
                verts = verts.reshape(-1, 3)
                # Homogeneous transform to world space — only Z row needed
                mw = np.array(obj.matrix_world, dtype=np.float64)
                world_z = mw[2, 0] * verts[:, 0] + mw[2, 1] * verts[:, 1] + mw[2, 2] * verts[:, 2] + mw[2, 3]
                global_min_z = min(global_min_z, float(world_z.min()))
                eval_obj.to_mesh_clear()
        else:
            # Fallback: bone endpoints
            for pb in target_armature.pose.bones:
                global_min_z = min(global_min_z,
                                   (target_armature.matrix_world @ pb.head).z,
                                   (target_armature.matrix_world @ pb.tail).z)

    if global_min_z == float("inf"):
        print("  Floor snap: nothing to snap, skipping")
        return

    z_delta = floor_z + FLOOR_SNAP_MARGIN - global_min_z
    print(f"  Floor snap: global_min={global_min_z:.3f}  floor={floor_z:.3f}  margin={FLOOR_SNAP_MARGIN}  correction={z_delta:+.4f}")

    if abs(z_delta) < 0.0005:
        return

    # Shift any object-level Z location keyframes on the armature.
    action = target_armature.animation_data.action if target_armature.animation_data else None
    if action:
        shifted = 0
        for fc in _get_action_fcurves(action):
            if fc.data_path == "location" and fc.array_index == 2:
                for kf in fc.keyframe_points:
                    kf.co.y += z_delta
                fc.update()
                shifted += 1
        print(f"  Floor snap: shifted {shifted} Z-location fcurve(s)")

    # Also update the live location so subsequent view_layer evaluations are correct.
    target_armature.location.z += z_delta


def orient_cameras_to_actor(cameras, actors_all_objs, frame_start):
    """
    Rotate each scene camera to point at the combined bounding-box centre of all
    actor meshes.  After rotating, matrix_world is refreshed so fit_cameras_to_actors
    can use it.
    """
    bpy.context.scene.frame_set(frame_start)
    bpy.context.view_layer.update()

    all_corners = []
    for model_objs in actors_all_objs:
        for obj in model_objs:
            if obj.type != "MESH":
                continue
            for corner in obj.bound_box:
                all_corners.append(obj.matrix_world @ Vector(corner))

    if not all_corners:
        print("orient_cameras_to_actor: no mesh objects found, skipping")
        return

    pts = np.array([[v.x, v.y, v.z] for v in all_corners])
    target_pt = Vector(pts.mean(axis=0).tolist())

    for cam in cameras:
        direction = target_pt - cam.location
        if direction.length < 0.001:
            continue
        cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        print(f"  {cam.name}: aimed at ({target_pt.x:.2f}, {target_pt.y:.2f}, {target_pt.z:.2f})")

    # Refresh so fit_cameras_to_actors reads updated camera matrix_world.
    bpy.context.view_layer.update()


def render_reference_image(output_dir, target_armature, actors_all_objs):
    """
    Renders a single front-facing reference image of the model in rest pose.
    A temporary camera is created, fitted to the character, and removed after.
    Saved as reference.png in output_dir.
    Only called for single-actor jobs.
    """
    import mathutils

    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Switch to rest pose by temporarily clearing the baked action ──
    prev_frame = scene.frame_current
    anim_data  = target_armature.animation_data
    prev_action = anim_data.action if anim_data else None
    if anim_data:
        anim_data.action = None
    for pb in target_armature.pose.bones:
        pb.matrix_basis = mathutils.Matrix.Identity(4)
    bpy.context.view_layer.update()

    # ── 2. Compute character bounding box and forward direction ───────────
    all_corners = []
    for obj in actors_all_objs:
        if obj.type != "MESH":
            continue
        for corner in obj.bound_box:
            all_corners.append(obj.matrix_world @ Vector(corner))

    if not all_corners:
        # Fallback to armature location if no meshes found
        all_corners = [target_armature.location.copy()]

    pts    = np.array([[v.x, v.y, v.z] for v in all_corners])
    center = Vector(pts.mean(axis=0).tolist())
    height = float(pts[:, 2].max() - pts[:, 2].min())

    # Character's local -Y is the forward-facing direction in world space
    fwd = (target_armature.matrix_world.to_3x3() @ Vector((0.0, -1.0, 0.0))).normalized()

    # Place camera 1.5× character height in front, at centre height
    cam_distance = max(height * 1.5, 1.0)
    cam_location = center + fwd * cam_distance
    cam_location.z = center.z

    # ── 3. Create and aim temporary camera ───────────────────────────────
    cam_data = bpy.data.cameras.new("__ref_cam__")
    cam_obj  = bpy.data.objects.new("__ref_cam__", cam_data)
    scene.collection.objects.link(cam_obj)
    cam_obj.location = cam_location
    direction = center - cam_location
    cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    bpy.context.view_layer.update()

    # Fit focal length using the same logic as fit_cameras_to_actors
    fit_cameras_to_actors([cam_obj], [actors_all_objs], 0, RENDER_WIDTH, RENDER_HEIGHT)
    bpy.context.view_layer.update()

    # ── 4. Render single frame ────────────────────────────────────────────
    prev_camera        = scene.camera
    scene.camera       = cam_obj
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = os.path.join(output_dir, "reference")
    bpy.ops.render.render(write_still=True)
    print(f"Reference image saved -> {output_dir}/reference.png")

    # ── 5. Restore state and remove temporary camera ──────────────────────
    scene.camera = prev_camera
    if anim_data and prev_action:
        anim_data.action = prev_action
    scene.frame_set(prev_frame)
    bpy.context.view_layer.update()

    bpy.data.objects.remove(cam_obj)
    bpy.data.cameras.remove(cam_data)


def cleanup_objects(objects):
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        if obj.name in bpy.data.objects:
            obj.select_set(True)
    bpy.ops.object.delete()

    try:
        bpy.ops.outliner.orphans_purge(do_recursive=True)
    except Exception:
        pass


def clear_pose_constraints(armature):
    for bone in armature.pose.bones:
        for constraint in list(bone.constraints):
            bone.constraints.remove(constraint)


def clear_animation(armature):
    if armature.animation_data:
        armature.animation_data_clear()


def main():
    import time
    # Apply render resolution from CLI args (--width / --height)
    scene.render.resolution_x          = RENDER_WIDTH
    scene.render.resolution_y          = RENDER_HEIGHT
    scene.render.resolution_percentage = 100

    cameras = get_scene_cameras()
    if MAX_CAMERAS is not None:
        cameras = cameras[:MAX_CAMERAS]
        print(f"Limiting to {MAX_CAMERAS} camera(s) for this run.")
    spawn_objs = get_spawn_locations()

    if spawn_objs:
        print(f"Found {len(spawn_objs)} spawn point(s): {[o.name for o in spawn_objs]}")
    else:
        print("No spawn points found in scene; using X-axis offset fallback")

    actors_data = []      # list of (model_file, anim_file, target_armature, canonical_bones)
    actors_all_objs = []  # list of model_objs per actor for cleanup

    global_frame_start = None
    global_frame_end   = None

    for actor_idx, (model_file, anim_file) in enumerate(ACTORS):
        actor_t0 = time.time()
        print(f"\n=== Loading actor {actor_idx}: model={Path(model_file).name}, anim={Path(anim_file).name} ===")

        # Check for bone world coords output files and skip calculation if present
        target_bone_log_path = os.path.join(OUTPUT_ROOT, "target_bone_world_coords.txt")
        source_bone_log_path = os.path.join(OUTPUT_ROOT, "source_bone_world_coords.txt")
        skip_bone_export = False
        if os.path.exists(target_bone_log_path) and os.path.exists(source_bone_log_path):
            print(f"[SKIP] {target_bone_log_path} and {source_bone_log_path} already exist, skipping bone world coords export.")
            skip_bone_export = True

        t0 = time.time()

        t_import_model_start = time.time()
        model_objs = import_file(model_file)
        t_import_model_end = time.time()
        print(f"  [TIMER] Model import: {t_import_model_end-t_import_model_start:.2f}s")
        target_armature = find_first_armature(model_objs)
        add_corrective_smooth(model_objs)
        t_corrective_end = time.time()
        print(f"  [TIMER] Corrective smooth: {t_corrective_end-t_import_model_end:.2f}s")

        anim_objs = import_file(anim_file)
        t_import_anim_end = time.time()
        print(f"  [TIMER] Anim import: {t_import_anim_end-t_corrective_end:.2f}s")
        source_armature = find_first_armature(anim_objs)
        t_pre_constraint_end = time.time()
        print(f"  [TIMER] Pre-constraint update: {t_pre_constraint_end-t_import_anim_end:.2f}s")

        # Place actor at its scene spawn point, or fall back to X-axis offset.
        # IMPORTANT: create_constraints() drives target_armature.location each frame
        # from src_root_world (source armature world position), so only the SOURCE
        # armature's position determines where the actor ends up in the world.
        # The target armature Z must also match spawn Z so that tgt_armature_init_z
        # is set to the scene floor level inside create_constraints().
        spawn_floor_z = 0.0
        if actor_idx < len(spawn_objs):
            spawn = spawn_objs[actor_idx]
            spawn_loc = spawn.matrix_world.translation
            spawn_floor_z = spawn_loc.z
            spawn_rot_z = spawn.matrix_world.to_euler().z
            source_armature.location.x       = spawn_loc.x
            source_armature.location.y       = spawn_loc.y
            source_armature.location.z       = spawn_loc.z
            source_armature.rotation_euler.z = spawn_rot_z
            target_armature.location.x       = spawn_loc.x
            target_armature.location.y       = spawn_loc.y
            target_armature.location.z       = spawn_loc.z
            print(f"  Spawn: {spawn.name}  "
                  f"loc=({spawn_loc.x:.2f}, {spawn_loc.y:.2f}, {spawn_loc.z:.2f})  "
                  f"rot_z={math.degrees(spawn_rot_z):.1f}°")
        elif actor_idx > 0:
            # Fallback: both source AND target must be offset; create_constraints drives
            # target.location from src_root_world so source position is what matters.
            offset_x = actor_idx * ACTOR_SPACING
            source_armature.location.x += offset_x
            target_armature.location.x += offset_x
            print(f"  Spawn: X-offset fallback +{offset_x:.1f} m")

        # Force a depsgraph update so matrix_world reflects the new locations before
        # create_constraints() reads armature_basis from matrix_world.
        bpy.context.view_layer.update()

        clear_pose_constraints(target_armature)
        clear_animation(target_armature)
        create_constraints(source_armature, target_armature)
        add_shoulder_limits(target_armature)
        t_constraints_end = time.time()
        print(f"  [TIMER] Constraints & limits: {t_constraints_end-t_pre_constraint_end:.2f}s")

        frame_start, frame_end = get_frame_range(source_armature)

        # Use the intersection of all actors' frame ranges
        if global_frame_start is None:
            global_frame_start, global_frame_end = frame_start, frame_end
        else:
            global_frame_start = max(global_frame_start, frame_start)
            global_frame_end   = min(global_frame_end,   frame_end)

        bake_to_target(target_armature, frame_start, frame_end)
        t_bake_end = time.time()
        print(f"  [TIMER] Animation baking: {t_bake_end-t_constraints_end:.2f}s")

        canonical_bones = resolve_canonical_bones(target_armature)
        t_canonical_end = time.time()
        print(f"  [TIMER] Canonical bones: {t_canonical_end-t_bake_end:.2f}s")
        print(f"  Snapping actor {actor_idx} to floor (floor_z={spawn_floor_z:.3f})")
        apply_floor_snap(target_armature, canonical_bones, model_objs, spawn_floor_z, frame_start, frame_end)
        t_floor_snap_end = time.time()
        print(f"  [TIMER] Floor snap: {t_floor_snap_end-t_canonical_end:.2f}s")


        # Only skip bone world coords export if requested
        if not skip_bone_export:
            # ...existing code for bone world coords export (if any) would go here...
            pass

        cleanup_objects(anim_objs)
        t_cleanup_end = time.time()
        print(f"  [TIMER] Cleanup anim objs: {t_cleanup_end-t_floor_snap_end:.2f}s")

        actors_data.append((model_file, anim_file, target_armature, canonical_bones))
        actors_all_objs.append(model_objs)

        print(f"Resolved canonical bones for actor {actor_idx}:")
        print(f"  [TIMER] Total actor setup: {time.time()-actor_t0:.2f}s")
        for k in CANONICAL_JOINTS:
            if k in canonical_bones:
                print(f"  {k:>15} -> {canonical_bones[k].name}")
            else:
                print(f"  {k:>15} -> (missing)")

    frame_start = global_frame_start
    frame_end   = global_frame_end
    scene.frame_start = frame_start
    scene.frame_end   = frame_end

    print(f"\nScene:      {Path(bpy.data.filepath).name}")
    print(f"Actors:     {len(ACTORS)}")
    print(f"Frame range:{frame_start} -> {frame_end}")

    print("\nOrienting cameras to actors...")
    orient_cameras_to_actor(cameras, actors_all_objs, frame_start)

    print("\nFitting cameras to actors...")
    fit_cameras_to_actors(cameras, actors_all_objs, frame_start, RENDER_WIDTH, RENDER_HEIGHT)

    job_root = Path(OUTPUT_ROOT)

    if len(ACTORS) == 1:
        _, _, target_armature, _ = actors_data[0]
        render_reference_image(str(job_root), target_armature, actors_all_objs[0])

    for cam in cameras:
        cam_dir = job_root / cam.name
        render_camera_sequence(cam, str(cam_dir), actors_data, frame_start, frame_end)

    for model_objs in actors_all_objs:
        cleanup_objects(model_objs)

    print("Done.")


if __name__ == "__main__":
    # if not PIL_AVAILABLE:
    #     raise RuntimeError(
    #         "Overlay rendering requires Pillow in Blender's Python. "
    #         "Install it with Blender's bundled python.exe using: python -m pip install pillow"
    #     )
    try:
        main()
    except Exception as e:
        import traceback
        error_log_path = os.path.join(OUTPUT_ROOT, "error_log.txt")
        with open(error_log_path, "w", encoding="utf-8") as f:
            f.write("Exception occurred:\n")
            traceback.print_exc(file=f)
        raise