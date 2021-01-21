import time, os
import logging
import streamlit as st
import numpy as np
from PIL import Image
from settings import IMAGE_DIR, DURATION, WAVE_OUTPUT_FILE
from src.sound import sound
from src.model import CNN
from setup_logging import setup_logging

setup_logging()
logger = logging.getLogger('app')

def init_model():
    cnn = CNN((128, 87))
    cnn.load_model()
    return cnn


def main():
    title = "Guitar Chord Recognition"
    st.title(title)
    image = Image.open(os.path.join(IMAGE_DIR, 'app_guitar.jpg'))
    st.image(image, use_column_width=True)

    if st.button('Record'):
        with st.spinner(f'Recording for {DURATION} seconds ....'):
            sound.record()
        st.success("Recording completed")

    if st.button('Play'):
        # sound.play()
        try:
            audio_file = open(WAVE_OUTPUT_FILE, 'rb')
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/wav')
        except:
            st.write("Please record sound first")



if __name__ == '__main__':
    main()
    # for i in range(100):
    #   # Update the progress bar with each iteration.
    #   latest_iteration.text(f'Iteration {i+1}')
    #   bar.progress(i + 1)
    #   time.sleep(0.1)

