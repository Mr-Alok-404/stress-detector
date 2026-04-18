# Use a stable Python environment
FROM python:3.11

# 1. Create the specific non-root user required by Hugging Face (UID 1000)
RUN useradd -m -u 1000 user

# 2. Set up environment variables for the new user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# 3. Set the working directory to the user's home folder
WORKDIR $HOME/app

# 4. Copy requirements and install them
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy all project files and give ownership to the new user
COPY --chown=user . .

# 6. Switch to the non-root user
USER user

# 7. Expose the port Hugging Face requires
EXPOSE 7860

# 8. Start the Gunicorn production server
CMD ["gunicorn", "-b", "0.0.0.0:7860", "--timeout", "120", "app:app"]
