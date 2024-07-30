# Use the official Python image
FROM python:3.10-slim

# Set environment variables to ensure Python output is sent straight to terminal (e.g. logs)
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file first to leverage Docker cache
COPY requirements.txt .

# Install required Python packages with a default timeout
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Copy the rest of the application files to the container's working directory
COPY . .

# Expose the port the application runs on
EXPOSE 5000

# Define the command to run the application
CMD ["gunicorn", "app:app"]

