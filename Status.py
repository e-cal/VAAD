from time import sleep


class Status:
    def __init__(self):
        self.recording = False
        self.ready = True
        self.audio = False
        self.prev_res = None

    def start_recording(self):
        self.recording = True
        self.ready = False

    def stop_recording(self):
        self.recording = False
        sleep(3)
        self.ready = True
