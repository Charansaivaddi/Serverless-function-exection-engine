FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the function code into the container
COPY . /app

# Install any dependencies (if required)
RUN pip install --no-cache-dir -r requirements.txt || true

# Command to run the Python script
CMD ["python", "main.py"]
