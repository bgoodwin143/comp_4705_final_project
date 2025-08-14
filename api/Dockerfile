# Start from a slim, official Python base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy only the requirements file first to leverage Docker's caching mechanism
COPY ./api/requirements.txt /app/requirements.txt

# Install the Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire API directory into the container
COPY ./api /app

# Expose the port that the API will run on
EXPOSE 8000

# Define the command to run the application using uvicorn
# The --host 0.0.0.0 makes it accessible from outside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]