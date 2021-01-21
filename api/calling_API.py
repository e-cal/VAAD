from google.cloud import dialogflow

# link for integrating with API:
# https://cloud.google.com/dialogflow/es/docs/quick/api

# Still working on finishing this, mostly just copied and pasted
# the code from the link and started working through it

def detect_intent_texts(project_id, session_id, texts, language_code):
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

if __name__ == "__main__":
    # These should all be the same
    project_id = "vaad-302015"
    # Session ID can be whatever and it's stored for like 20 min to keep a record
    # But we can also change this if needed but this one should work!
    session_id = 123456789
    language_code = "en-US"
    # texts is the input, so I think it can be either a string, text input or audio
    # This should be the only variable that changes I think
    texts = " What is QMIND"
    detect_intent_texts(project_id, session_id, texts, language_code)