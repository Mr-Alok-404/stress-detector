# Use a stable Python environment
FROM python:3.11

# Set the working directory
WORKDIR /code

# Copy your requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your project files into the server
COPY . .

# Hugging Face requires web apps to run on port 7860
EXPOSE 7860

# Start the Gunicorn production server
CMD ["gunicorn", "-b", "0.0.0.0:7860", "--timeout", "120", "app:app"]
