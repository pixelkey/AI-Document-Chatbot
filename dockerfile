FROM python:3.11.8-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir --timeout=120 -r requirements.txt


# Expose the port
EXPOSE 7860

# Run the application
CMD ["python", "scripts/main.py"]
