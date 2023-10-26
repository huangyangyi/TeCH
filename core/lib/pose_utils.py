eps = 0.01
JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "jaw"
]

SMPLX_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "jaw",
    "left_eye_smplx",
    "right_eye_smplx",
    "left_index1",
    "left_index2",
    "left_index3",
    "left_middle1",
    "left_middle2",
    "left_middle3",
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_ring1",
    "left_ring2",
    "left_ring3",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "right_index1",
    "right_index2",
    "right_index3",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
    "right_eye_brow1",
    "right_eye_brow2",
    "right_eye_brow3",
    "right_eye_brow4",
    "right_eye_brow5",
    "left_eye_brow5",
    "left_eye_brow4",
    "left_eye_brow3",
    "left_eye_brow2",
    "left_eye_brow1",
    "nose1",
    "nose2",
    "nose3",
    "nose4",
    "right_nose_2",
    "right_nose_1",
    "nose_middle",
    "left_nose_1",
    "left_nose_2",
    "right_eye1",
    "right_eye2",
    "right_eye3",
    "right_eye4",
    "right_eye5",
    "right_eye6",
    "left_eye4",
    "left_eye3",
    "left_eye2",
    "left_eye1",
    "left_eye6",
    "left_eye5",
    "right_mouth_1",
    "right_mouth_2",
    "right_mouth_3",
    "mouth_top",
    "left_mouth_3",
    "left_mouth_2",
    "left_mouth_1",
    "left_mouth_5",
    "left_mouth_4",
    "mouth_bottom",
    "right_mouth_4",
    "right_mouth_5",
    "right_lip_1",
    "right_lip_2",
    "lip_top",
    "left_lip_2",
    "left_lip_1",
    "left_lip_3",
    "lip_bottom",
    "right_lip_3",
    "right_contour_1",
    "right_contour_2",
    "right_contour_3",
    "right_contour_4",
    "right_contour_5",
    "right_contour_6",
    "right_contour_7",
    "right_contour_8",
    "contour_middle",
    "left_contour_8",
    "left_contour_7",
    "left_contour_6",
    "left_contour_5",
    "left_contour_4",
    "left_contour_3",
    "left_contour_2",
    "left_contour_1",
    "head_top",
    "left_big_toe",
    "left_ear",
    "left_eye",
    "left_heel",
    "left_index",
    "left_middle",
    "left_pinky",
    "left_ring",
    "left_small_toe",
    "left_thumb",
    "nose",
    "right_big_toe",
    "right_ear",
    "right_eye",
    "right_heel",
    "right_index",
    "right_middle",
    "right_pinky",
    "right_ring",
    "right_small_toe",
    "right_thumb",
]

OPENPOSE_NAMES = [
    "nose", 
    "neck", 
    "right_shoulder", 
    "right_elbow", 
    "right_wrist", 
    "left_shoulder", 
    "left_elbow", 
    "left_wrist", 
    "pelvis", 
    "right_hip", 
    "right_knee", 
    "right_ankle", 
    "left_hip", 
    "left_knee", 
    "left_ankle", 
    "right_eye", 
    "left_eye", 
    "right_ear", 
    "left_ear", 
    "left_big_toe", 
    "left_small_toe", 
    "left_heel", 
    "right_big_toe", 
    "right_small_toe", 
    "right_heel", 
    "left_wrist", 
    "left_thumb1", 
    "left_thumb2", 
    "left_thumb3", 
    "left_thumb", 
    "left_index1", 
    "left_index2", 
    "left_index3", 
    "left_index", 
    "left_middle1", 
    "left_middle2", 
    "left_middle3", 
    "left_middle", 
    "left_ring1", 
    "left_ring2", 
    "left_ring3", 
    "left_ring", 
    "left_pinky1", 
    "left_pinky2", 
    "left_pinky3", 
    "left_pinky", 
    "right_wrist", 
    "right_thumb1", 
    "right_thumb2", 
    "right_thumb3", 
    "right_thumb", 
    "right_index1", 
    "right_index2", 
    "right_index3", 
    "right_index", 
    "right_middle1", 
    "right_middle2", 
    "right_middle3", 
    "right_middle", 
    "right_ring1", 
    "right_ring2", 
    "right_ring3", 
    "right_ring", 
    "right_pinky1", 
    "right_pinky2", 
    "right_pinky3", 
    "right_pinky", 
    "right_eye_brow1", 
    "right_eye_brow2", 
    "right_eye_brow3", 
    "right_eye_brow4", 
    "right_eye_brow5", 
    "left_eye_brow5", 
    "left_eye_brow4", 
    "left_eye_brow3", 
    "left_eye_brow2", 
    "left_eye_brow1", 
    "nose1", 
    "nose2", 
    "nose3", 
    "nose4", 
    "right_nose_2", 
    "right_nose_1", 
    "nose_middle", 
    "left_nose_1", 
    "left_nose_2", 
    "right_eye1", 
    "right_eye2", 
    "right_eye3", 
    "right_eye4", 
    "right_eye5", 
    "right_eye6", 
    "left_eye4", 
    "left_eye3", 
    "left_eye2", 
    "left_eye1", 
    "left_eye6", 
    "left_eye5", 
    "right_mouth_1", 
    "right_mouth_2", 
    "right_mouth_3", 
    "mouth_top", 
    "left_mouth_3", 
    "left_mouth_2", 
    "left_mouth_1", 
    "left_mouth_5", 
    "left_mouth_4", 
    "mouth_bottom", 
    "right_mouth_4", 
    "right_mouth_5", 
    "right_lip_1", 
    "right_lip_2", 
    "lip_top", 
    "left_lip_2", 
    "left_lip_1", 
    "left_lip_3", 
    "lip_bottom", 
    "right_lip_3", 
    "right_contour_1", 
    "right_contour_2", 
    "right_contour_3", 
    "right_contour_4", 
    "right_contour_5", 
    "right_contour_6", 
    "right_contour_7", 
    "right_contour_8", 
    "contour_middle", 
    "left_contour_8", 
    "left_contour_7", 
    "left_contour_6", 
    "left_contour_5", 
    "left_contour_4", 
    "left_contour_3", 
    "left_contour_2", 
    "left_contour_1"
]

OPENPOSE_BODY = [
    "nose",
    "neck",
    "right_shoulder", 
    "right_elbow", 
    "right_wrist", 
    "left_shoulder", 
    "left_elbow", 
    "left_wrist", 
    "right_hip", 
    "right_knee", 
    "right_ankle", 
    "left_hip", 
    "left_knee", 
    "left_ankle", 
    "right_eye", 
    "left_eye", 
    "right_ear", 
    "left_ear", 
]

OPENPOSE_LEFT_HAND = [
    "left_wrist", 
    "left_thumb1", 
    "left_thumb2", 
    "left_thumb3", 
    "left_thumb", 
    "left_index1", 
    "left_index2", 
    "left_index3", 
    "left_index", 
    "left_middle1", 
    "left_middle2", 
    "left_middle3", 
    "left_middle", 
    "left_ring1", 
    "left_ring2", 
    "left_ring3", 
    "left_ring", 
    "left_pinky1", 
    "left_pinky2", 
    "left_pinky3", 
    "left_pinky", 
]

OPENPOSE_RIGHT_HAND = [
    "right_wrist", 
    "right_thumb1", 
    "right_thumb2", 
    "right_thumb3", 
    "right_thumb", 
    "right_index1", 
    "right_index2", 
    "right_index3", 
    "right_index", 
    "right_middle1", 
    "right_middle2", 
    "right_middle3", 
    "right_middle", 
    "right_ring1", 
    "right_ring2", 
    "right_ring3", 
    "right_ring", 
    "right_pinky1", 
    "right_pinky2", 
    "right_pinky3", 
    "right_pinky", 
]

OPENPOSE_FACE = [
    "right_eye_brow1", 
    "right_eye_brow2", 
    "right_eye_brow3", 
    "right_eye_brow4", 
    "right_eye_brow5", 
    "left_eye_brow5", 
    "left_eye_brow4", 
    "left_eye_brow3", 
    "left_eye_brow2", 
    "left_eye_brow1", 
    "nose1", 
    "nose2", 
    "nose3", 
    "nose4", 
    "right_nose_2", 
    "right_nose_1", 
    "nose_middle", 
    "left_nose_1", 
    "left_nose_2", 
    "right_eye1", 
    "right_eye2", 
    "right_eye3", 
    "right_eye4", 
    "right_eye5", 
    "right_eye6", 
    "left_eye4", 
    "left_eye3", 
    "left_eye2", 
    "left_eye1", 
    "left_eye6", 
    "left_eye5", 
    "right_mouth_1", 
    "right_mouth_2", 
    "right_mouth_3", 
    "mouth_top", 
    "left_mouth_3", 
    "left_mouth_2", 
    "left_mouth_1", 
    "left_mouth_5", 
    "left_mouth_4", 
    "mouth_bottom", 
    "right_mouth_4", 
    "right_mouth_5", 
    "right_lip_1", 
    "right_lip_2", 
    "lip_top", 
    "left_lip_2", 
    "left_lip_1", 
    "left_lip_3", 
    "lip_bottom", 
    "right_lip_3", 
    "right_contour_1", 
    "right_contour_2", 
    "right_contour_3", 
    "right_contour_4", 
    "right_contour_5", 
    "right_contour_6", 
    "right_contour_7", 
    "right_contour_8", 
    "contour_middle", 
    "left_contour_8", 
    "left_contour_7", 
    "left_contour_6", 
    "left_contour_5", 
    "left_contour_4", 
    "left_contour_3", 
    "left_contour_2", 
    "left_contour_1"
]

import cv2
import numpy as np
import math 
import matplotlib

def draw_bodypose(canvas, candidate):
    H, W, C = canvas.shape
    candidate = np.array(candidate)

    stickwidth = 4

    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    for i in range(17):
        index = np.array(limbSeq[i]) - 1
        Y = candidate[index.astype(int), 0] * float(W)
        X = candidate[index.astype(int), 1] * float(H)
        if X[0] < eps or X[1] < eps:
            continue
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, polygon, colors[i])

    canvas = (canvas * 0.6).astype(np.uint8)

    for i in range(18):
        x, y = candidate[i][0:2]
        x = int(x * W)
        y = int(y * H)
        if x > eps and y > eps:
            cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)

    return canvas


def draw_handpose(canvas, peaks):
    H, W, C = canvas.shape

    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

    peaks = np.array(peaks)

    for ie, e in enumerate(edges):
        x1, y1 = peaks[e[0]]
        x2, y2 = peaks[e[1]]
        x1 = int(x1 * W)
        y1 = int(y1 * H)
        x2 = int(x2 * W)
        y2 = int(y2 * H)
        if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
            cv2.line(canvas, (x1, y1), (x2, y2), matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0]) * 255, thickness=2)

    for i, keyponit in enumerate(peaks):
        x, y = keyponit
        x = int(x * W)
        y = int(y * H)
        if x > eps and y > eps:
            cv2.circle(canvas, (x, y), 4, (0, 0, 255), thickness=-1)
    return canvas


def draw_facepose(canvas, lmks):
    H, W, C = canvas.shape
    lmks = np.array(lmks)
    for lmk in lmks:
        x, y = lmk
        x = int(x * W)
        y = int(y * H)
        if x > eps and y > eps:
            cv2.circle(canvas, (x, y), 3, (255, 255, 255), thickness=-1)
    return canvas

def draw_openpose_map(canvas, smplx_keypoints_2d, smplx_keypoints_mask):
    assert len(SMPLX_NAMES) == len(smplx_keypoints_2d)
    kp_dict = dict()
    for k, p, b in zip(SMPLX_NAMES, smplx_keypoints_2d, smplx_keypoints_mask):
        b = b or k in JOINT_NAMES
        if not b:
            p *= 0
        kp_dict[k] = p
    body_points = [kp_dict[k] for k in OPENPOSE_BODY]
    left_hand_points = [kp_dict[k] for k in OPENPOSE_LEFT_HAND]
    right_hand_points = [kp_dict[k] for k in OPENPOSE_RIGHT_HAND]
    face_points = [kp_dict[k] for k in OPENPOSE_FACE]
    draw_bodypose(canvas, body_points)
    draw_handpose(canvas, left_hand_points)
    draw_handpose(canvas, right_hand_points)
    draw_facepose(canvas, face_points)
    return canvas