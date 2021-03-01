from itertools import product
from random import randint

import pandas as pd
import numpy as np
import mediapipe as mp

from model.utils import compute_distance, get_mean

HANDMARK = mp.solutions.hands.HandLandmark
POSEMARK = mp.solutions.pose.PoseLandmark

FINGERS = ["THUMB", "INDEX_FINGER", "MIDDLE_FINGER", "RING_FINGER", "PINKY"]
CARTESIAN_FINGERS = [('THUMB', 'INDEX_FINGER'), ('THUMB', 'MIDDLE_FINGER'), ('THUMB', 'RING_FINGER'), ('THUMB', 'PINKY'),
                     ('MIDDLE_FINGER', 'INDEX_FINGER'), ('MIDDLE_FINGER', 'RING_FINGER'),
                     ('MIDDLE_FINGER', 'PINKY'), ('PINKY', 'INDEX_FINGER'), ('PINKY', 'RING_FINGER'),
                     ('RING_FINGER', 'INDEX_FINGER')]


class DataframeLandmark:

    def __init__(self, nb_frames=15):
        self.nb_frames = nb_frames
        self.cols = self.get_col_df_list()
        self.rows = []
        self.tmp_cols = []

    def __len__(self):
        return len(self.rows)

    def get_col_df_list(self):
        cols = []

        for finger_a, finger_b in CARTESIAN_FINGERS:
            cols.append(f"l_hand_dist_{finger_a}_{finger_b}")
        for finger in FINGERS:
            cols.append(f"l_hand_dist_WRIST_{finger}")
        for finger_a, finger_b in CARTESIAN_FINGERS:
            cols.append(f"r_hand_dist_{finger_a}_{finger_b}")
        for finger in FINGERS:
            cols.append(f"r_hand_dist_WRIST_{finger}")
        # relative distance with origin mean HEAD
        for cpt in range(0, 21):
            cols.append(f"l_hand_x_{cpt}")
            cols.append(f"l_hand_y_{cpt}")
            cols.append(f"l_hand_z_{cpt}")
        for cpt in range(0, 21):
            cols.append(f"r_hand_x_{cpt}")
            cols.append(f"r_hand_y_{cpt}")
            cols.append(f"r_hand_z_{cpt}")
        for cpt in range(0, 25):
            cols.append(f"pose_x_{cpt}")
            cols.append(f"pose_y_{cpt}")
            cols.append(f"pose_z_{cpt}")
        return cols

    def append_landmarks(self, results_hand, results_pose):
        row = []
        tmp_row = []
        # hand landmarks process results_hands.multi_hand_landmarks[0].ListFields()[0][1][20].x
        hands = [hand.label.lower() for hand in results_hand.multi_handedness[0].ListFields()[0][1]]
        pose_points = results_pose.pose_landmarks.ListFields()[0][1]
        landmarks = dict(zip(hands, results_hand.multi_hand_landmarks))

        mean_head = get_mean(
            np.array([pose_points[POSEMARK.RIGHT_EYE_INNER].x, pose_points[POSEMARK.RIGHT_EYE_INNER].y, pose_points[POSEMARK.RIGHT_EYE_INNER].z]),
            np.array([pose_points[POSEMARK.LEFT_EYE_INNER].x, pose_points[POSEMARK.LEFT_EYE_INNER].y, pose_points[POSEMARK.LEFT_EYE_INNER].z]),
            np.array([pose_points[POSEMARK.MOUTH_LEFT].x, pose_points[POSEMARK.MOUTH_LEFT].y,
                     pose_points[POSEMARK.MOUTH_LEFT].z]),
            np.array([pose_points[POSEMARK.MOUTH_RIGHT].x, pose_points[POSEMARK.MOUTH_RIGHT].y,
                     pose_points[POSEMARK.MOUTH_RIGHT].z])
        )

        if landmarks.get("left", False):
            hand_points = landmarks["left"].ListFields()[0][1]
            wirst_point = np.array([hand_points[HANDMARK.WRIST].x, hand_points[HANDMARK.WRIST].y, hand_points[HANDMARK.WRIST].z])
            # COMPUTE DISTANCE BETWEEN FINGER TIPS
            for finger_a, finger_b in CARTESIAN_FINGERS:
                point_a = np.array([hand_points[HANDMARK[f"{finger_a}_TIP"]].x, hand_points[HANDMARK[f"{finger_a}_TIP"]].y,
                                    hand_points[HANDMARK[f"{finger_a}_TIP"]].z])
                point_b = np.array([hand_points[HANDMARK[f"{finger_b}_TIP"]].x, hand_points[HANDMARK[f"{finger_b}_TIP"]].y,
                                    hand_points[HANDMARK[f"{finger_b}_TIP"]].z])
                dist = compute_distance(point_a, point_b)
                row.append(dist)

            # COMPUTE WRIST DISTANCE
            for finger in FINGERS:
                finger_tip = np.array([hand_points[HANDMARK[f"{finger}_TIP"]].x, hand_points[HANDMARK[f"{finger}_TIP"]].y,
                                      hand_points[HANDMARK[f"{finger}_TIP"]].z])
                row.append(compute_distance(finger_tip, wirst_point))

            # ADD RELATIVE COORDINATE FROM MEAN HEAD
            for landmark in landmarks["left"].ListFields()[0][1]:
                row += [landmark.x - mean_head[0], landmark.y - mean_head[1], landmark.z - mean_head[2]]

        else:
            row += np.zeros(15 + 3 * 21).tolist()

        if landmarks.get("right", False):
            hand_points = landmarks["right"].ListFields()[0][1]
            wirst_point = np.array(
                [hand_points[HANDMARK.WRIST].x, hand_points[HANDMARK.WRIST].y, hand_points[HANDMARK.WRIST].z])
            # COMPUTE DISTANCE BETWEEN FINGER TIPS
            for finger_a, finger_b in CARTESIAN_FINGERS:
                point_a = np.array(
                    [hand_points[HANDMARK[f"{finger_a}_TIP"]].x, hand_points[HANDMARK[f"{finger_a}_TIP"]].y,
                     hand_points[HANDMARK[f"{finger_a}_TIP"]].z])
                point_b = np.array(
                    [hand_points[HANDMARK[f"{finger_b}_TIP"]].x, hand_points[HANDMARK[f"{finger_b}_TIP"]].y,
                     hand_points[HANDMARK[f"{finger_b}_TIP"]].z])
                dist = compute_distance(point_a, point_b)
                row.append(dist)

            # COMPUTE WRIST DISTANCE
            for finger in FINGERS:
                finger_tip = np.array(
                    [hand_points[HANDMARK[f"{finger}_TIP"]].x, hand_points[HANDMARK[f"{finger}_TIP"]].y,
                     hand_points[HANDMARK[f"{finger}_TIP"]].z])
                row.append(compute_distance(finger_tip, wirst_point))

            # ADD RELATIVE COORDINATE FROM MEAN HEAD
            for landmark in landmarks["right"].ListFields()[0][1]:
                row += [landmark.x - mean_head[0], landmark.y - mean_head[1], landmark.z - mean_head[2]]
        else:
            row += np.zeros(15 + 3 * 21).tolist()

        # pose landmarks process
        for landmark in pose_points:
            row += [landmark.x, landmark.y, landmark.z]

        self.rows.append(row)
        self.tmp_cols.append(tmp_row)

    def get_dataframe(self):
        if len(self.rows) < self.nb_frames:
            cpt = 0
            while len(self.rows) < self.nb_frames:
                idx = cpt % len(self.rows)
                mean_row = [(value[0] + value[1])/2 for value in zip(self.rows[idx], self.rows[idx + 1])]
                self.rows = self.rows[:idx] + [mean_row] + self.rows[idx:]
                cpt += 2
        elif len(self.rows) > self.nb_frames:
            cpt = 0
            while len(self.rows) > self.nb_frames:
                idx = cpt % len(self.rows)
                del self.rows[idx + 1]
                cpt += 1
        columns = [f'{col}_{row}' for col, row in product(self.cols, range(0, self.nb_frames))]
        df_rows = []
        for row in self.rows:
            df_rows += row
        return pd.DataFrame([df_rows], columns=columns)
