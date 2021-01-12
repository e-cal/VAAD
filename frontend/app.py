import urllib
import os
import sys
import importlib.util
import cv2
import streamlit as st


def module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    st.write("""
             # Hi Kavin
             Say *cheese!*
             """)
    Detector = module_from_file("detector", "../attn_detection/detector.py")
    detector = Detector.FaceDetector()
    detector.run()


if __name__ == "__main__":
    main()
