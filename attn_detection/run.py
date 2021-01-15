from detector import FaceDetector


def main():
    detector = FaceDetector()
    detector.run()
#    prob = detector.detect()
#    if prob == None:
#        print("No face detected")
#        return
#
#    print(f"face detected with {(prob * 100):.2f}% certainty")
    prob = detector.detect()
    if not prob:
        print("No face detected")
        return

    print(f"face detected with {(prob * 100):.2f}% certainty")


main()
