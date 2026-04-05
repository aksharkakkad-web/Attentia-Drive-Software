"""Quick MediaPipe FaceLandmarker smoke test.

Opens the webcam, runs FaceLandmarker on one frame, prints whether a face
was detected plus head pose angles, then closes the webcam.

Run: python check_mediapipe.py
"""

import sys
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = "models/face_landmarker.task"


def main() -> None:
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )

    with mp_vision.FaceLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR: Could not open webcam")
            sys.exit(1)

        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            print("ERROR: Could not read frame from webcam")
            sys.exit(1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        if not result.face_landmarks:
            print("No face detected in frame.")
            return

        print(f"Face detected: {len(result.face_landmarks)} face(s)")

        if result.facial_transformation_matrixes:
            import numpy as np
            mat = np.array(result.facial_transformation_matrixes[0].data).reshape(4, 4)
            # Extract yaw, pitch, roll from rotation matrix
            r00, r10, r20 = mat[0, 0], mat[1, 0], mat[2, 0]
            r21, r22 = mat[2, 1], mat[2, 2]
            r01, r02 = mat[0, 1], mat[0, 2]
            pitch = np.degrees(np.arctan2(-r20, np.sqrt(r00**2 + r10**2)))
            yaw = np.degrees(np.arctan2(r10, r00))
            roll = np.degrees(np.arctan2(r21, r22))
            print(f"Head pose — Pitch: {pitch:.1f}°  Yaw: {yaw:.1f}°  Roll: {roll:.1f}°")
        else:
            print("Head pose: transformation matrix not available")


if __name__ == "__main__":
    main()
