<<<<<<< HEAD
# Use an official Node.js runtime as base image
FROM node:18-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the function code into the container (optional if needed at build time)
COPY . .

# Default command (will be overridden in your exec)
CMD ["node"]

=======
FROM node:16-slim

# Set the working directory
WORKDIR /app

# Copy the function code into the container
COPY . /app

# Install any dependencies (if required)
RUN npm install || true

# Command to run the JavaScript file
CMD ["node", "main.js"]
>>>>>>> origin/main
