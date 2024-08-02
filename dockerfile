# Use official Python 3.11.8 image from DockerHub
FROM python:3.11.8-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 7860 available to the world outside this container
EXPOSE 7860

# Define environment variable
ENV NAME Chatbot

# Run chatbot.py when the container launches
CMD ["python", "scripts/chatbot.py"]