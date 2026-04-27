FROM python:3.10

# Hugging Face Spaces require running on port 7860
ENV PORT=7860

# Install necessary system libraries for OpenCV
USER root
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Set up a new user named "user" with user ID 1000 (Required by Hugging Face)
RUN useradd -m -u 1000 user

# Set home to the user's home directory
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Copy the requirements file first to leverage Docker cache
COPY --chown=user ./requirements.txt $HOME/app/requirements.txt

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY --chown=user . $HOME/app

# Switch to the non-root user
USER user

# Start the FastAPI application via uvicorn
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
