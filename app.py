# pylint: disable=no-member
import time
import importlib.util
import sys
import threading
import sounddevice as sd
import streamlit as st
import cv2
from record import record_to_file
import api
import Assistant as assistant
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


def main():
    st.title("VAAD")

    cap = get_cap()
    detector = get_detector()
    classifier = get_classifier()
    status = get_status()

    run = st.checkbox("Run")
    show_cam = st.checkbox("Show camera feed", value=True)
    listen_stat = st.empty()
    res_audio_container = st.empty()
    res_text_container = st.empty()
    cam_container = st.empty()
    overlay = None
    # TODO fix responses disappearing when disabling run

    frame_count = 0
    while True:
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if status.audio:
            res_audio, res_text = assistant.detect_intent_audio("query.wav")
            res_audio_container.audio(res_audio, format='audio/mp3')
            res_text_container.text(res_text)
            status.audio = False

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
                frame = overlay

        if not ret:
            print("Something went wrong, cam died.")
            cap.release()
            break

        if show_cam:
            cam_container.image(frame, channels="RGB")
        else:
            cam_container.image("transparent.png", channels="RGB")


if __name__ == "__main__":
    main()
