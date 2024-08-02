# Navigate to your project directory
cd support-chatbot

# Create a virtual environment using Python 3.11.8
python3.11 -m venv venv

# Activate the virtual environment
source venv/bin/activate

pip install -r requirements.txt

python scripts/chatbot.py