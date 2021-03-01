# pylint: disable=no-member
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
import assistant
from Status import Status
from model.detector import FaceDetector  # pylint: disable=wrong-import-position
from model.classifier import AttentionClassifier  # pylint: disable=wrong-import-position


@st.cache(allow_output_mutation=True)
def get_cap():
    return cv2.VideoCapture(0)


@st.cache(allow_output_mutation=True)
def get_detector():
    return FaceDetector()


@st.cache(allow_output_mutation=True)
def get_classifier():
    return AttentionClassifier("model/attention_model.pth")


@st.cache(allow_output_mutation=True)
def get_status():
    return Status()


def send_to_assistant():
    project_id = "vaad-302015"
    session_id = 123456789
    language_code = "en-US"
    texts = ["What is QMIND"]
    api.detect_intent_texts(project_id, session_id, texts,
                            language_code)


def run_record(status):
    record_to_file('query.wav')
    status.audio = True
    status.stop_recording()


async def get_response():
    res = await assistant.detect_intent_audio("query.wav")
    return res


async def main():
    st.title("VAAD")

    cap = get_cap()
    detector = get_detector()
    classifier = get_classifier()
    status = get_status()

    run = st.checkbox("Run")
    listen_stat = st.empty()
    res_container = st.empty()
    frameST = st.empty()
    overlay = None

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if status.audio:
            res = await get_response()

            # with open("query.wav", 'rb') as query:
            #     audio = query.read()
            #     res_container.audio(audio, format='audio/wav')
            # status.audio = False

        if run:
            frame_count += 1
            if status.recording:
                listen_stat.text("Listening.")
            else:
                listen_stat.text("Done.")

            if frame_count % 2 == 0:
                overlay = detector.overlay(frame)

            if not overlay is None:
                label = classifier.classify(
                    frame, ['attentive', 'inattentive'])
                attentive = label == 'attentive'
                if attentive:
                    # Start recording audio (if recording isn't in progress)
                    if status.ready:
                        status.start_recording()
                        record_thread = threading.Thread(
                            target=run_record,
                            args=(status,)
                        )
                        record_thread.start()
                classifier.overlay(overlay, label)
                # overlay facebox
                frame = cv2.cvtColor(
                    overlay, cv2.COLOR_BGR2RGB)

        if not ret:
            print("Something went wrong, cam died.")
            cap.release()
            break

        frameST.image(frame, channels="RGB")

if __name__ == "__main__":
    main()
