import time
from utils.menu import *
import cv2
from detector import FaceDetector


def capture(label: str):
    cam = cv2.VideoCapture(0)
    detector = FaceDetector(cam)
    while True:
        _, frame = cam.read()
        cv2.imshow(label, frame)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        if key == ord(" "):
            print("Checking for face...")
            if detector.detect(frame):
                print("Face detected, capturing image...")
                img_name = f"data/{label}/{str(int(time.time() * 1000))}.png"
                print(f"{label} image saved to: {img_name}")
                cv2.imwrite(img_name, frame)
            else:
                print("No face detected.")
    cam.release()
    cv2.destroyAllWindows()


def auto_capture(label: str):
    cam = cv2.VideoCapture(0)
    detector = FaceDetector(cam)
    while True:
        _, frame = cam.read()
        cv2.imshow(label, frame)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        time.sleep(0.1)
        print("Checking for face...")
        if detector.detect(frame):
            print("Face detected, capturing image...")
            img_name = f"data/{label}/{str(int(time.time() * 1000))}.png"
            print(f"{label} image saved to: {img_name}")
            cv2.imwrite(img_name, frame)
        else:
            print("No face detected.")
    cam.release()
    cv2.destroyAllWindows()


def main():
    modes = ["user", "automatic"]
    mode = menu(modes, "Method of gathering data:",
                "Choose a data gathering method: ")
    labels = ["attentive", "inattentive"]
    label_idx = menu(labels, "Label:",
                     "Enter the label you would like to add data for: ")
    label = labels[label_idx]
    print(f"Capturing data for label: {label}")
    if mode == 0:
        capture(label)
    else:
        auto_capture(label)


main()
