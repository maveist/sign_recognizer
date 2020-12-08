import argparse

import mediapipe as mp
import cv2

import displayer
from parsermedia.camera import CameraStream
from parsermedia.video import VideoStream
from model.trainers.trainer import train_model_from_videos

# TODO movement detection (feu vert feu rouge)
# TODO donnee entrainement simple "bonjour", "oui", "non", "arbre", "bebe"


def run():
    pass


if __name__ == "__main__":
    # TODO Change the behavior of the command. By default it should display the evaluation because it is
    # what the users want to use when they dsicover the projet.
    # Instead of having an argument --evaluate to launch the evaluation you should launch it by default
    # and add an argument '--no-evaluation' to use when you just want to display the structure of the video.
    parser = argparse.ArgumentParser(description='Sign to text: Command that parse a video stream and recognizes signs')
    parser.add_argument("-v", "--video", type=str, nargs='?')
    parser.add_argument("-c", '--camera', action="store_true")
    parser.add_argument("-t", '--train', action="store_true")
    parser.add_argument("-e", "--evaluate", action="store_true")
    args = parser.parse_args()

    # init stream value
    stream = None
    if not (args.process or args.upload):
        parser.error('No stream given, add --video or --camera')
    if args.video:
        stream = VideoStream(args.video)
    if args.camera:
        stream = CameraStream()

    # init components
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose

    # launch action
    if args.evaluate:
        displayer.display_evaluate_from_stream(stream, mp_pose, mp_hands)
    if args.train:
        train_model_from_videos()
    else:
        displayer.display_from_stream(stream, mp_pose, mp_hands)
