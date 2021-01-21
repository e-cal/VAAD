import importlib.util
import api
import cv2
import streamlit as st
import sounddevice as sd
from scipy.io.wavfile import write


def module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


Detector = module_from_file("detector", "../attn_detection/detector.py")


@st.cache(allow_output_mutation=True)
def get_cap():
    return cv2.VideoCapture(0)


@st.cache(allow_output_mutation=True)
def get_alt_cap():
    return cv2.VideoCapture(2)


@st.cache(allow_output_mutation=True)
def get_detector():
    return Detector.FaceDetector()


if __name__ == "__main__":
    st.title("Test")
    alt = st.checkbox("Use alternate webcam")
    record = st.button("Use Audio Recorder")

    if alt:
        cap = get_alt_cap()
    else:
        cap = get_cap()

    if record:
        fs = 44100  # Sample rate
        seconds = 5  # Duration of recording

        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
        sd.wait()  # Wait until recording is finished
        write('output.wav', fs, myrecording)  # Save as WAV file

    detector = get_detector()

    run = st.checkbox("Run")
    frameST = st.empty()

    while run:
        ret, frame = cap.read()
        overlay = detector.overlay(frame)
        if not overlay is None:
            project_id = "vaad-302015"
            session_id = 123456789
            language_code = "en-US"
            texts = ["What is QMIND"]
            api.detect_intent_texts(
                project_id, session_id, texts, language_code)
            frame = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        # Stop the program if reached end of video
        if not ret:
            print("Done processing !!!")
            cv2.waitKey(3000)
            # Release device
            cap.release()
            break

        frameST.image(frame, channels="RGB")
