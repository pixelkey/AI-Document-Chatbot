# scripts/interface.py

import gradio as gr
from chatbot_functions import chatbot_response, clear_history

def setup_gradio_interface(context):
    """
    Sets up the Gradio interface.
    Args:
        context (dict): Context containing client, memory, and other settings.
    Returns:
        gr.Blocks: Gradio interface object.
    """
    with gr.Blocks() as app:
        # Output fields
        chat_history = gr.Textbox(label="Chat History", lines=10, interactive=False)
        references = gr.Textbox(label="References", lines=2, interactive=False)

        # Input fields
        input_text = gr.Textbox(label="Input Text")
        submit_button = gr.Button("Submit")
        clear_button = gr.Button("Clear")

        # Setup event handlers
        submit_button.click(
            lambda input_text: chatbot_response(input_text, context),
            inputs=input_text,
            outputs=[chat_history, references, input_text]
        )
        clear_button.click(lambda: clear_history(context), outputs=[chat_history, references, input_text])

        input_text.submit(
            lambda input_text: chatbot_response(input_text, context),
            inputs=input_text,
            outputs=[chat_history, references, input_text]
        )

        # Layout
        gr.Column([chat_history, references, input_text, submit_button, clear_button])

    return app
