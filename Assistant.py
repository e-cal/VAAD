import wave
from google.cloud import dialogflow

project_id = "vaad-302015"
session_id = 1
language_code = "en-US"


def detect_intent_texts(texts):
    """Returns the result of detect intent with texts as inputs.

    Using the same `session_id` between requests allows continuation
    of the conversation."""
    session_client = dialogflow.SessionsClient()

    session = session_client.session_path(project_id, session_id)
    print('Session path: {}\n'.format(session))

    for text in texts:
        text_input = dialogflow.TextInput(
            text=text, language_code=language_code)

        query_input = dialogflow.QueryInput(text=text_input)

        response = session_client.detect_intent(
            request={'session': session, 'query_input': query_input})

        print('=' * 20)
        print('Query text: {}'.format(response.query_result.query_text))
        print('Detected intent: {} (confidence: {})\n'.format(
            response.query_result.intent.display_name,
            response.query_result.intent_detection_confidence))
        print('Fulfillment text: {}\n'.format(
            response.query_result.fulfillment_text))


def detect_intent_audio(audio_file_path):
    """Returns the result of detect intent with an audio file as input.

    Using the same `session_id` between requests allows continuation
    of the conversation."""

    session_client = dialogflow.SessionsClient()

    # Note: hard coding audio_encoding and sample_rate_hertz for simplicity.
    audio_encoding = dialogflow.AudioEncoding.AUDIO_ENCODING_LINEAR_16
    sample_rate_hertz = 44100

    session = session_client.session_path(project_id, session_id)
    print('Session path: {}\n'.format(session))

    with open(audio_file_path, 'rb') as audio_file:
        input_audio = audio_file.read()

    audio_config = dialogflow.InputAudioConfig(
        audio_encoding=audio_encoding, language_code=language_code,
        sample_rate_hertz=sample_rate_hertz)
    query_input = dialogflow.QueryInput(audio_config=audio_config)

    request = dialogflow.DetectIntentRequest(
        session=session,
        query_input=query_input,
        input_audio=input_audio,
    )
    response = session_client.detect_intent(request=request)

    print('=' * 20)
    print('Query text: {}'.format(response.query_result.query_text))
    print('Detected intent: {} (confidence: {})\n'.format(
        response.query_result.intent.display_name,
        response.query_result.intent_detection_confidence))
    print('Fulfillment text: {}\n'.format(
        response.query_result.fulfillment_text))
    # print(response)
    with open('response.mp3', 'wb') as f:
        f.write(response.output_audio)


def detect_intent_stream(audio_file_path):
    """Returns the result of detect intent with streaming audio as input.

    Using the same `session_id` between requests allows continuation
    of the conversation."""
    session_client = dialogflow.SessionsClient()

    # Note: hard coding audio_encoding and sample_rate_hertz for simplicity.
    audio_encoding = dialogflow.AudioEncoding.AUDIO_ENCODING_LINEAR_16
    sample_rate_hertz = 44100

    session_path = session_client.session_path(project_id, session_id)
    print('Session path: {}\n'.format(session_path))

    def request_generator(audio_config, audio_file_path):
        query_input = dialogflow.QueryInput(audio_config=audio_config)

        # The first request contains the configuration.
        yield dialogflow.StreamingDetectIntentRequest(
            session=session_path, query_input=query_input)

        # Here we are reading small chunks of audio data from a local
        # audio file.  In practice these chunks should come from
        # an audio input device.
        with open(audio_file_path, 'rb') as audio_file:
            while True:
                chunk = audio_file.read(4096)
                if not chunk:
                    break
                # The later requests contains audio data.
                yield dialogflow.StreamingDetectIntentRequest(input_audio=chunk)

    audio_config = dialogflow.InputAudioConfig(
        audio_encoding=audio_encoding, language_code=language_code,
        sample_rate_hertz=sample_rate_hertz)

    requests = request_generator(audio_config, audio_file_path)
    responses = session_client.streaming_detect_intent(requests=requests)

    print('=' * 20)
    for response in responses:
        print('Intermediate transcript: "{}".'.format(
            response.recognition_result.transcript))

    # Note: The result from the last response is the final transcript along
    # with the detected content.
    query_result = response.query_result
    print('=' * 20)
    print('Query text: {}'.format(query_result.query_text))
    print('Detected intent: {} (confidence: {})\n'.format(
        query_result.intent.display_name,
        query_result.intent_detection_confidence))
    print('Fulfillment text: {}\n'.format(
        query_result.fulfillment_text))


if __name__ == "__main__":
    texts = ["What is QMIND"]
    audio_file = "query.wav"
    detect_intent_audio(audio_file)
