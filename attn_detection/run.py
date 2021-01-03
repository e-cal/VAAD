from detector import FaceDetector


def main():
    detector = FaceDetector()
    prob = detector.detect()
    if prob == None:
        prob = 0
        print("No face detected")
    else:
        prob = prob[0]
        print(f"face detected with {(prob * 100):.2f}% certainty")


main()
