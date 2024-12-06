# Use the official Python image as the base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file
# COPY requirements.txt .

# Install dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
# COPY . .

# Expose the port that the Flask app runs on
EXPOSE 5000

# Command to run the Flask app
CMD ["flask", "run", "--host=0.0.0.0"]
