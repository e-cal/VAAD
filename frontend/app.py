import time
import asyncio
import importlib.util
import sys
import threading
import sounddevice as sd
import streamlit as st
import cv2
import api
from record import record_to_file
sys.path.append("../")
sys.path.append("./")
import model.detector as Detector  # pylint: disable=wrong-import-position


@st.cache(allow_output_mutation=True)
def get_cap():
    return cv2.VideoCapture(0)  # pylint: disable=no-member


@st.cache(allow_output_mutation=True)
def get_alt_cap():
    return cv2.VideoCapture(2)  # pylint: disable=no-member


@st.cache(allow_output_mutation=True)
def get_detector():
    return Detector.FaceDetector()


def send_to_assistant():
    project_id = "vaad-302015"
    session_id = 123456789
    language_code = "en-US"
    texts = ["What is QMIND"]
    api.detect_intent_texts(project_id, session_id, texts,  # pylint: disable=no-member
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
        record_to_file('demo.wav')
        print("done")
        done_thread = threading.Thread(target=finish_recording)
        done_thread.start()
    else:
        print("still recording...")


if __name__ == "__main__":
    st.title("Test")
    alt = st.checkbox("Use alternate webcam")
    if alt:
        cap = get_alt_cap()
    else:
        cap = get_cap()

    detector = get_detector()

    run = st.checkbox("Run")
    frameST = st.empty()

    while run:
        # when webcam is running, run record to file function
        ret, frame = cap.read()
        overlay = detector.overlay(frame)
        if not overlay is None:
            frame = cv2.cvtColor(  # pylint: disable=no-member
                overlay, cv2.COLOR_BGR2RGB)  # pylint: disable=no-member

            # this could be bad
            record_thread = threading.Thread(target=run_record)
            record_thread.start()

        # Stop the program if reached end of video
        if not ret:
            print("Camera stopped recording")
            cv2.waitKey(3000)  # pylint: disable=no-member
            # Release device
            cap.release()
            break

        frameST.image(frame, channels="RGB")
