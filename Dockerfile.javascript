FROM node:16-slim

# Set the working directory
WORKDIR /app

# Copy the function code into the container
COPY . /app

# Install any dependencies (if required)
RUN npm install || true

# Command to run the JavaScript file
CMD ["node", "main.js"]
