import argparse

import mediapipe as mp
import cv2

import displayer
from parsermedia.camera import CameraStream
from parsermedia.video import VideoStream
from model.trainers.trainer import train_model_from_videos

# TODO movement detection (feu vert feu rouge)
# TODO donnee entrainement simple "bonjour", "oui", "non", "arbre", "bebe"

def run_simple():
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        min_detection_confidence=0.7, min_tracking_confidence=0.5)
    pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        results_pose = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            print(results.multi_hand_landmarks)
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        if results_pose.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    hands.close()
    pose.close()
    cap.release()


def run():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sign to text')
    parser.add_argument("-v", "--video", type=str, nargs='?')
    parser.add_argument("-c", '--camera', action="store_true")
    parser.add_argument("-t", '--train', action="store_true")
    parser.add_argument("-e", "--evaluate", action="store_true")
    args = parser.parse_args()

    if args.evaluate:
        if args.video:
            stream = VideoStream(args.video)
            mp_hands = mp.solutions.hands
            mp_pose = mp.solutions.pose

            displayer.display_evaluate_from_stream(stream, mp_pose, mp_hands)
        if args.camera:
            stream = CameraStream()
            mp_hands = mp.solutions.hands
            mp_pose = mp.solutions.pose

            displayer.display_evaluate_from_stream(stream, mp_pose, mp_hands)
    else:
        if args.video:
            stream = VideoStream(args.video)
            mp_hands = mp.solutions.hands
            mp_pose = mp.solutions.pose

            displayer.display_from_stream(stream, mp_pose, mp_hands)
        if args.camera:
            stream = CameraStream()
            mp_hands = mp.solutions.hands
            mp_pose = mp.solutions.pose

            displayer.display_from_stream(stream, mp_pose, mp_hands)

        if args.train:
            train_model_from_videos()
