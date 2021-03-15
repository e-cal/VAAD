# Virtual Assistant Attention Detection
This project enables a user to interact with a virtual assistant by simply
looking at the device and speaking to it. No more unnatural activation phrases!
The app is powered by an MTCNN face recognition model, and a CNN classifier
trained to identify when the user is paying attention to the device. When the
model determines that the user is paying attention, it begins listening and
sends the user's request to a custom Dialogflow agent. The response is provided
as both audio from the assistant, and a plaintext transcript.

This work was presented at the *Canadian Undergraduate Conference on Artificial
Intelligence* 2021. This work was done with *QMIND - Queen's AI Hub*.

## Running the project
1. Install the dependencies from `requirements.txt`
2. Unzip `model/model.zip`
3. From the project's root directory, run `streamlit run app.py`
