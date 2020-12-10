import argparse

import mediapipe as mp
import cv2

import displayer
from parsermedia.camera import CameraStream
from parsermedia.video import VideoStream
from model.trainers.trainer import train_model_from_videos

# TODO movement detection (feu vert feu rouge)


def run():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sign to text: Command that parse a video stream and recognizes signs')
    parser.add_argument("-v", "--video", type=str, nargs='?')
    parser.add_argument("-t", '--train', action="store_true")
    parser.add_argument("-n", "--no-evaluate", action="store_true")
    args = parser.parse_args()

    # init stream value
    stream = CameraStream()

    if args.video:
        stream = VideoStream(args.video)

    # init components
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose

    # launch action
    if args.no_evaluate:
        displayer.display_from_stream(stream, mp_pose, mp_hands)
    if args.train:
        train_model_from_videos()
    else:
        displayer.display_evaluate_from_stream(stream, mp_pose, mp_hands)
