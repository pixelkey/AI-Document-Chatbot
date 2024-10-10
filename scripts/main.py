# scripts/main.py

from initialize import initialize_model_and_retrieval
from interface import setup_gradio_interface

def main():
    # Initialize the model, embeddings, and retrieval components
    context = initialize_model_and_retrieval()

    # Setup and launch Gradio interface
    app = setup_gradio_interface(context)
    app.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main()
