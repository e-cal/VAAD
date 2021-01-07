from utils.menu import *
import time
import cv2


def capture(label: str):
    cam = cv2.VideoCapture(0)
    while True:
        ret, frame = cam.read()
        cv2.imshow(label, frame)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        # elif key % 256 == 32:
        elif key == ord(" "):
            imName = f"images/{label}/{str(int(time.time() * 1000))}.png"
            print(f"{label} image saved.")
            cv2.imwrite(imName, frame)
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
