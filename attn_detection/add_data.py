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
                print("Face detected, capturing image...")
                img_name = f"images/{label}/{str(int(time.time() * 1000))}.png"
                print(f"{label} image saved to: {img_name}")
                cv2.imwrite(img_name, frame)
            else:
                print("No face detected.")
    cam.release()
    cv2.destroyAllWindows()


def main():
    labels = ["attentive", "inattentive"]
    label_idx = menu(labels, "Label:",
                     "Enter the label you would like to add data for: ")
    label = labels[label_idx]
    print(f"Capturing data for label: {label}")
    capture(label)


main()
