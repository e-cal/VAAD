import time
import importlib.util
import sys
import threading
import sounddevice as sd
import streamlit as st
import cv2
from record import record_to_file
import Assistant as assistant
from Status import Status
from model.detector import FaceDetector
from model.classifier import AttentionClassifier


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


def run_record(status):
    record_to_file('query.wav')
    status.audio = True
    status.stop_recording()


def main():
    st.title("Interact with Vlad the VAAD")
    st.text("A virtual assistant that pays attention.")

    cap = get_cap()
    detector = get_detector()
    classifier = get_classifier()
    status = get_status()

    run = st.sidebar.checkbox("Run")
    show_cam = st.sidebar.checkbox("Show camera", value=True)
    listen_stat = st.empty()
    if status.prev_res is not None:
        st.header("Response")
        res_audio_container = st.audio(status.prev_res[0], format='audio/mp3')
        st.subheader("Transcript")
        res_text_container = st.text(status.prev_res[1])
    else:
        res_audio_container = st.empty()
        res_text_container = st.empty()

    cam_container = st.empty()
    overlay = None

    frame_count = 0
    attentive_time = 0
    while True:
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if status.audio:
            res_audio, res_text = assistant.detect_intent_audio("query.wav")
            res_audio_container.audio(res_audio, format='audio/mp3')
            res_text_container.text(res_text)
            status.audio = False
            status.prev_res = (res_audio, res_text)

        if run:
            frame_count += 1
            if status.recording:
                listen_stat.subheader("Listening...")
            else:
                listen_stat.text("")

            if frame_count % 2 == 0:
                overlay = detector.overlay(frame)

            if not overlay is None:
                label = classifier.classify(
                    frame, ['attentive', 'inattentive'])
                attentive = label == 'attentive'
                if attentive:
                    attentive_time += 1
                    if status.ready and attentive_time > 5:
                        status.start_recording()
                        record_thread = threading.Thread(
                            target=run_record,
                            args=(status,)
                        )
                        record_thread.start()
                else:
                    attentive_time = 0
                classifier.overlay(overlay, label)
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
