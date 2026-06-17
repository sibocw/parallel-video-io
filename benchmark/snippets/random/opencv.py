# Random-access decode with OpenCV (POS_FRAMES seek).
import cv2


def get_frames(path, indices):
    cap = cv2.VideoCapture(path)
    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames
