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

        # Initialize session state separately for each user
        session_state = gr.State(value=[])

        # Setup event handlers with explicit state management
        submit_button.click(
            lambda input_text, history: chatbot_response(input_text, context, history),
            inputs=[input_text, session_state],
            outputs=[chat_history, references, input_text, session_state]
        )
        clear_button.click(
            lambda history: clear_history(context, history),
            inputs=[session_state],
            outputs=[chat_history, references, input_text, session_state]
        )

        input_text.submit(
            lambda input_text, history: chatbot_response(input_text, context, history),
            inputs=[input_text, session_state],
            outputs=[chat_history, references, input_text, session_state]
        )

        # Layout
        gr.Column([chat_history, references, input_text, submit_button, clear_button])

    return app
