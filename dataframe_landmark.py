from itertools import product

from random import randint

import pandas as pd
import numpy as np


class DataframeLandmark:

    def __init__(self):
        self.cols = self.get_col_df_list()
        self.rows = []

    def __len__(self):
        return len(self.rows)

    def get_col_df_list(self):
        cols = []
        for cpt in range(0, 21):
            cols.append(f"l_hand_x_{cpt}")
            cols.append(f"l_hand_y_{cpt}")
            cols.append(f"l_hand_z_{cpt}")
        for cpt in range(0,21):
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
        # hand landmarks process results_hands.multi_hand_landmarks[0].ListFields()[0][1][20].x
        hands = [hand.label.lower() for hand in results_hand.multi_handedness[0].ListFields()[0][1]]
        landmarks = dict(zip(hands, results_hand.multi_hand_landmarks))

        if landmarks.get("left", False):
            for landmark in landmarks["left"].ListFields()[0][1]:
                row += [landmark.x, landmark.y, landmark.z]
        else:
            row += np.zeros(3*21).tolist()

        if landmarks.get("right", False):
            for landmark in landmarks["right"].ListFields()[0][1]:
                row += [landmark.x, landmark.y, landmark.z]
        else:
            row += np.zeros(3*21).tolist()

        # pose landmarks process
        for landmark in results_pose.pose_landmarks.ListFields()[0][1]:
            row += [landmark.x, landmark.y, landmark.z]
        self.rows.append(row)


    def get_random_dataframe_with_target_value(self):
        if len(self.rows) > 0:
            columns = [f'{col}_{row}' for col, row in product(self.cols, range(0, 20))]
            dataframe_rows = []
            for i in range(0,51):
                picked_indexes = []
                for k in range(0, min(len(self.rows), 20)):  # We randomly pick data point of 26 images and we do it 50 times
                    good_pick = False
                    while not good_pick:
                        picked_idx = randint(0, len(self.rows)-1)
                        if picked_idx not in picked_indexes:
                            picked_indexes.append(picked_idx)
                            good_pick = True
                picked_indexes.sort()
                row = []
                for idx in picked_indexes:
                    row += self.rows[idx]
                # fill void cell by None
                for j in range(0,(20-len(picked_indexes))*len(self.cols)):
                    row.append(None)
                dataframe_rows.append(row)
            return pd.DataFrame(dataframe_rows, columns=columns)
        else:
            return None


