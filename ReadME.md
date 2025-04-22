# Project Documentation: AWS Lambda Replica

**Last Updated:** 09-04-2025

---

## Getting Started

Instructions for setting up and running the project for the first time.

### Prerequisites

List all software, tools, accounts, or permissions required before installation.

*   Python
*   Node.js
*   Docker

### Installation

Provide step-by-step instructions to install the project and its dependencies.

```bash
# Example Installation Steps
git clone [repository-url]
cd [project-directory]
pip install -r requirements.txt
```

### Running the Project
- **Backend:**  
  Start the FastAPI server:
  ```bash
  uvicorn main:app --reload
  ```
- **Frontend:**  
  Open the `index.html` file in your browser to access the dashboard.

---

## Usage

### Basic Usage
- Use the dashboard for creating, editing, and deleting functions.
- Execute functions via the provided API endpoints.

### Advanced Usage
- Customize function execution by modifying timeout, language, etc.
- Integrate with external systems via REST API calls.

### Examples
- Example 1: Creating a new function.
- Example 2: Updating a function and observing changes in the Docker container execution.

---

### Key Components
- **Backend:** FastAPI, SQLAlchemy
- **Frontend:** HTML, CSS, JavaScript
- **Containers:** Docker for function isolation
- **Database:** SQLite

### Technology Stack
- Python, JavaScript, Docker, GitHub Actions

### Design Decisions
- Use Docker to encapsulate function execution.
- RESTful API for easy integration and scalability.

---

## Configuration

### Configuration Files
- `requirements.txt`: List of Python dependencies.
- Dockerfiles for Python and JavaScript functions.
- GitHub Actions workflow for CI/CD.

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

### Data Models
- **Function Model:** Includes name, route, language, timeout, and code_path.

---

## Troubleshooting

### Common Issues
- Docker not running: Ensure Docker is installed and started.
- API endpoints not responding: Check server logs and network configurations.

---
