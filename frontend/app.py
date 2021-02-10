import importlib.util
import cv2
import streamlit as st

#new imports
from sys import byteorder
from array import array
from struct import pack

import pyaudio
import wave

import time

THRESHOLD = 500
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 44100

#audio detection functions
def is_silent(snd_data):
    "Returns 'True' if below the 'silent' threshold"
    return max(snd_data) < THRESHOLD

def normalize(snd_data):
    "Average the volume out"
    MAXIMUM = 16384
    times = float(MAXIMUM)/max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i*times))
    return r

def trim(snd_data):
    "Trim the blank spots at the start and end"
    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i)>THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    # Trim to the left
    snd_data = _trim(snd_data)

    # Trim to the right
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data

def add_silence(snd_data, seconds):
    "Add silence to the start and end of 'snd_data' of length 'seconds' (float)"
    silence = [0] * int(seconds * RATE)
    r = array('h', silence)
    r.extend(snd_data)
    r.extend(silence)
    return r

def record():
    """
    Record a word or words from the microphone and 
    return the data as an array of signed shorts.

    Normalizes the audio, trims silence from the 
    start and end, and pads with 0.5 seconds of 
    blank sound to make sure VLC et al can play 
    it without getting chopped off.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
        input=True, output=True,
        frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False

    r = array('h')

    while 1:
        # little endian, signed short
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        silent = is_silent(snd_data)

        #if the recording started and it is silent, add 1 ms to num_silent
        if silent and snd_started:
            num_silent += 1
        #if the recording started and it is not silent again, set num_silent back to 0
        elif not silent and snd_started:
            num_silent = 0
        #if it is not silent and the recording is not started, start the recording
        elif not silent and not snd_started:
            snd_started = True
        #if the recording started and if it is silent for more than 1000*5 ms (5 seconds)
        if snd_started and num_silent > (1000*5):
            break

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    r = trim(r)
    r = add_silence(r, 0.5)
    return sample_width, r

def record_to_file(path):
    "Records from the microphone and outputs the resulting data to 'path'"
    sample_width, data = record()
    data = pack('<' + ('h'*len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()

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

    if alt:
        cap = get_alt_cap()
    else:
        cap = get_cap()


    detector = get_detector()

    run = st.checkbox("Run")
    frameST = st.empty()

    while run:
        #when webcam is running, run record to file function
        print("please speak a word into the microphone")
        record_to_file('demo.wav')
        print("done - result written to demo.wav")
        ret, frame = cap.read()
        overlay = detector.overlay(frame)
        if not overlay is None:
            frame = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        # Stop the program if reached end of video
        if not ret:
            print("Done processing !!!")
            cv2.waitKey(3000)
            # Release device
            cap.release()
            break

        frameST.image(frame, channels="RGB")
