from utils.menu import *
import time
import cv2
from detector import FaceDetector


def capture(label: str):
    cam = cv2.VideoCapture(0)
    detector = FaceDetector(cam)
    while True:
        ret, frame = cam.read()
        cv2.imshow(label, frame)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        elif key == ord(" "):
            print("Checking for face...")
            if detector.detect():
                print("face detected, capturing image...")
                imName = f"images/{label}/{str(int(time.time() * 1000))}.png"
                print(f"{label} image saved.")
                cv2.imwrite(imName, frame)
            else:
                print("No face detected.")
    cam.release()
    cv2.destroyAllWindows()


def main():
    labels = ["attentive", "inattentive"]
    labelIdx = menu(labels, "Label:",
                    "Enter the label you would like to add data for: ")
    label = labels[labelIdx]
    print(f"Capturing data for label: {label}")
    capture(label)


main()
