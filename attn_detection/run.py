from detector import *


def main():
    detector = FaceDetector()
    try:
        box, prob, ld = detector.detect()
        print(f"face detected with {(prob[0] * 100):.2f}% certainty")
    except:
        print("No face detected")


main()
