import time
import asyncio
import importlib.util
import sys
import threading
import sounddevice as sd
import streamlit as st
import cv2
from record import record_to_file
import api
# sys.path.append("./")
# import model.detector as Detector
from model.detector import FaceDetector  # pylint: disable=wrong-import-position
from model.classifier import AttentionClassifier  # pylint: disable=wrong-import-position

# pylint doesn't think cv2 has anything in it
# pylint: disable=no-member


@st.cache(allow_output_mutation=True)
def get_cap():
    return cv2.VideoCapture(0)


@st.cache(allow_output_mutation=True)
def get_alt_cap():
    return cv2.VideoCapture(2)


@st.cache(allow_output_mutation=True)
def get_detector():
    return FaceDetector()


@st.cache(allow_output_mutation=True)
def get_classifier():
    return AttentionClassifier("model/attention_model.pth")


def send_to_assistant():
    project_id = "vaad-302015"
    session_id = 123456789
    language_code = "en-US"
    texts = ["What is QMIND"]
    api.detect_intent_texts(project_id, session_id, texts,
                            language_code)


def check_recording():
    f = open("status", "r")
    status = f.read().strip()
    f.close()
    return status == "done"


def finish_recording():
    time.sleep(5)
    f = open("status", "w")
    f.write("done")
    f.close()


def run_record():
    run = check_recording()
    if run:
        print("recording...")
        f = open("status", "w")
        f.write("running")
        f.close()
        record_to_file('query.wav')
        print("done")
        done_thread = threading.Thread(target=finish_recording)
        done_thread.start()


if __name__ == "__main__":
    st.title("Test")
    alt = st.checkbox("Use alternate webcam")
    if alt:
        cap = get_alt_cap()
    else:
        cap = get_cap()

    detector = get_detector()
    classifier = get_classifier()

    run = st.checkbox("Run")
    frameST = st.empty()

    while run:
        # when webcam is running, run record to file function
        ret, frame = cap.read()
        overlay = detector.overlay(frame)
        if not overlay is None:
            # classify attention
            label = classifier.classify(frame, ['attentive', 'inattentive'])
            attentive = label == 'attentive'
            if attentive:
                # Start recording audio (if recording isn't in progress)
                record_thread = threading.Thread(target=run_record)
                record_thread.start()

            classifier.overlay(overlay, label)
            # overlay facebox
            frame = cv2.cvtColor(
                overlay, cv2.COLOR_BGR2RGB)

        # Stop the program if reached end of video
        if not ret:
            print("Camera stopped recording")
            cv2.waitKey(3000)
            # Release device
            cap.release()
            break

        frameST.image(frame, channels="RGB")
