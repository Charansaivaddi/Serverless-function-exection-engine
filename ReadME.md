# Project Documentation: Serverless Function Execution engine aka AWS Lambda Replica

**Last Updated:** 30-05-2025

---

### About Repo:
The Repository contains necessary code and documentation for Serverless function exection engine.
Serverless function exection engine is most similar to AWS Lambda. It allows function i.e., code execution with short burst time, and minimal cold-start time. Backend of the solution is adapted to solve the cold-start problem using a set of preloaded docker containers. Function met-data defines function name, burst time, langauge. Function codes are stored in an explicit file-storage.

## Getting Started

Instructions for setting up and running the project for the first time.

### Prerequisites

List all software, tools, accounts, or permissions required before installation.

*   Python
*   Node.js
*   Docker

### Installation Guide

Using python virtual environment (venv) is recommended to isolate the dependencies.
```
git clone [repository-url]
cd [project-directory]
pip install -r requirements.txt
```

### Running the Project
**Backend:**  
  Start the FastAPI server:
  ```
  uvicorn main:app --reload
  ```
**Frontend:**  
  Open the `index.html` file in your browser to access the dashboard.

---

## Usage

### Basic Usage
Use the dashboard for creating, editing, and deleting functions.
Execute functions via the provided API endpoints.
Customize function execution by modifying timeout, language(Python, JavaScript), etc.

### Examples
Example 1: Creating a new function.
Example 2: Updating a function and observing changes in the Docker container execution.

---

### Key Components
**Backend:** FastAPI, SQLAlchemy
**Frontend:** HTML, CSS, JavaScript
**Containers:** Docker for function isolation
**Database:** SQLite


---

## Configuration

### Configuration Files
`requirements.txt`: List of Python dependencies.
Dockerfiles for Python and JavaScript functions.

---

## API Reference

### Endpoints
- GET `/functions/` : List all functions.
- POST `/functions/` : Create a new function.
- GET `/functions/{id}` : Retrieve a specific function.
- PUT `/functions/{id}` : Update a function.
- DELETE `/functions/{id}` : Delete a function.
- POST `/functions/{id}/execute` : Execute a function.
- POST `/functions/execute` : Execute multiple functions.

---
