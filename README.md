# Virtual Assistant Attention Detection
This project enables a user to interact with a virtual assistant by simply
speaking to it (no activation phrase required)!
The app is powered by an MTCNN face recognition model, and a CNN classifier
trained to identify when the user is paying attention to the device. When the
model determines that the user is paying attention, it begins listening and
sends the user's request to a custom Dialogflow agent. The response is provided
as both audio from the assistant, and a plaintext transcript.
