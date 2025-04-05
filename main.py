import os
import shutil
import subprocess
import docker
import signal
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional, Dict
from db import Base, engine, SessionLocal, Function
from fastapi.middleware.cors import CORSMiddleware
from docker.errors import DockerException


# Create the database tables
Base.metadata.create_all(bind=engine)
app = FastAPI()
# Add this middleware after creating the app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this later to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Create code directory if it doesn't exist
CODE_DIR = "code"
os.makedirs(CODE_DIR, exist_ok=True)

# Dependency for DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic Schemas
class FunctionBase(BaseModel):
    name: str
    route: str
    language: str
    timeout: float
    code_path: str

class FunctionCreate(FunctionBase):
    code: str  # Actual code content

class FunctionUpdate(BaseModel):
    route: Optional[str] = None
    language: Optional[str] = None
    timeout: Optional[float] = None
    code: Optional[str] = None  # Updated code content

class FunctionOut(FunctionBase):
    id: int
    
    class Config:
        from_attributes = True

class ExecuteMultipleFunctionsRequest(BaseModel):
    function_ids: List[int]

def get_file_extension(language):
    """Get file extension based on language"""
    ext_map = {
        "python": ".py",
        "javascript": ".js"
    }
    return ext_map.get(language.lower(), ".txt")

def save_code_to_file(func_id, func_name, code_content, language):
    func_dir = os.path.join(CODE_DIR, str(func_id))
    os.makedirs(func_dir, exist_ok=True)
    ext = get_file_extension(language)
    # Change filename to always "main{ext}"
    filename = f"main{ext}"
    file_path = os.path.join(func_dir, filename)
    with open(file_path, "w") as f:
        f.write(code_content)
    return file_path

# Docker client
def initialize_docker_client():
    """Initialize the Docker client and handle errors gracefully."""
    try:
        client = docker.from_env()
        # Test Docker connection
        client.ping()
        return client
    except (DockerException, FileNotFoundError) as e:
        print("Warning: Docker is not available. Ensure Docker is running and accessible.")
        print(f"Error: {e}")
        return None

docker_client = initialize_docker_client()

def build_docker_image(func_id, func_name, language):
    """Build a Docker image for the function."""
    if not docker_client:
        raise HTTPException(status_code=500, detail="Docker is not available. Please start Docker.")
    
    func_dir = os.path.join(CODE_DIR, str(func_id))  # Directory for the function
    os.makedirs(func_dir, exist_ok=True)  # Ensure the directory exists
    
    dockerfile_template = f"Dockerfile.{language.lower()}"  # Template based on language
    dockerfile_path = os.path.join(func_dir, "Dockerfile")  # Destination Dockerfile path
    
    # Ensure the Dockerfile template exists
    if not os.path.exists(dockerfile_template):
        raise HTTPException(status_code=500, detail=f"Dockerfile template for {language} not found.")
    
    # Copy the Dockerfile template to the function's directory
    shutil.copy(dockerfile_template, dockerfile_path)
    
    # Removed renaming code as the function code is now stored as main{ext}
    image_tag = f"{func_name.lower()}_{func_id}"
    
    try:
        docker_client.images.build(path=func_dir, dockerfile="Dockerfile", tag=image_tag)
        return image_tag
    except docker.errors.BuildError as e:
        raise HTTPException(status_code=500, detail=f"Docker build failed: {str(e)}")

def run_function_in_docker(image_tag, timeout):
    """Run the function inside a Docker container with timeout enforcement."""
    if not docker_client:
        raise HTTPException(status_code=500, detail="Docker is not available. Please start Docker.")
    try:
        print(f"Running container for image: {image_tag} with timeout: {timeout}")
        container = docker_client.containers.run(image_tag, detach=True)
        print(f"Container {container.id} started.")
        
        result = container.wait(timeout=timeout)
        logs = container.logs().decode("utf-8")
        print(f"Container logs: {logs}")
        
        container.remove()
        print(f"Container {container.id} removed.")
        
        # Check for execution errors
        if result["StatusCode"] != 0:
            raise HTTPException(status_code=500, detail=f"Function execution failed with status code {result['StatusCode']}: {logs}")
        
        return logs
    except docker.errors.ContainerError as e:
        print(f"ContainerError: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Function execution failed: {str(e)}")
    except docker.errors.APIError as e:
        print(f"APIError: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Docker API error: {str(e)}")
    except subprocess.TimeoutExpired:
        print("TimeoutExpired: Function execution timed out")
        raise HTTPException(status_code=408, detail="Function execution timed out")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# Create a new function
@app.post("/functions/", response_model=FunctionOut)
def create_function(function: FunctionCreate, db: Session = Depends(get_db)):
    existing = db.query(Function).filter(Function.name == function.name).first()
    if existing:
        raise HTTPException(status_code=400, detail="Function with this name already exists")
    
    # Create function in database first to get ID
    db_function = Function(
        name=function.name,
        route=function.route,
        language=function.language,
        timeout=function.timeout,
        code_path=""  # Temporary, will update after saving file
    )
    
    db.add(db_function)
    db.commit()
    db.refresh(db_function)
    
    # Now save code to file using the function ID and name
    code_path = save_code_to_file(db_function.id, db_function.name, function.code, function.language)
    
    # Update the code_path in database
    db_function.code_path = code_path
    db.commit()
    db.refresh(db_function)
    
    return db_function

# Read all functions
@app.get("/functions/", response_model=List[FunctionOut])
def list_functions(db: Session = Depends(get_db)):
    return db.query(Function).all()

# Read a single function by ID
@app.get("/functions/{function_id}", response_model=FunctionOut)
def get_function(function_id: int, db: Session = Depends(get_db)):
    db_function = db.query(Function).filter(Function.id == function_id).first()
    if db_function is None:
        raise HTTPException(status_code=404, detail="Function not found")
    return db_function

# Get function code
@app.get("/functions/{function_id}/code")
def get_function_code(function_id: int, db: Session = Depends(get_db)):
    db_function = db.query(Function).filter(Function.id == function_id).first()
    if db_function is None:
        raise HTTPException(status_code=404, detail="Function not found")
    
    try:
        # Resolve absolute path from project directory based on stored path
        abs_path = db_function.code_path if os.path.isabs(db_function.code_path) else os.path.join(os.getcwd(), db_function.code_path)
        print("DEBUG: Resolved code file path:", abs_path)  # Debug logging
        
        # Fallback: if the resolved path does not exist, try using the standard naming (main{ext})
        if not os.path.exists(abs_path):
            ext = get_file_extension(db_function.language)
            fallback_path = os.path.join(os.getcwd(), "code", str(function_id), f"main{ext}")
            print("DEBUG: Fallback code file path:", fallback_path)
            abs_path = fallback_path
            if not os.path.exists(abs_path):
                raise HTTPException(status_code=404, detail=f"Code file not found at {abs_path}")
                
        with open(abs_path, "r", encoding="utf-8") as f:
            code_content = f.read()
        return {"name": db_function.name, "language": db_function.language, "code": code_content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error accessing code file: {str(e)}")

# Update a function by ID
@app.put("/functions/{function_id}", response_model=FunctionOut)
def update_function(function_id: int, function_update: FunctionUpdate, db: Session = Depends(get_db)):
    db_function = db.query(Function).filter(Function.id == function_id).first()
    if db_function is None:
        raise HTTPException(status_code=404, detail="Function not found")
    
    # Update database fields first
    update_data = function_update.dict(exclude_unset=True, exclude={"code"})
    for key, value in update_data.items():
        setattr(db_function, key, value)
    
    # Update code file if provided
    if function_update.code is not None:
        # If language was updated, we need the new language, otherwise use existing
        language = function_update.language if function_update.language else db_function.language
        
        # Save updated code to file
        code_path = save_code_to_file(function_id, db_function.name, function_update.code, language)
        db_function.code_path = code_path
    
    db.commit()
    db.refresh(db_function)
    return db_function

# Delete a function by ID
@app.delete("/functions/{function_id}")
def delete_function(function_id: int, db: Session = Depends(get_db)):
    db_function = db.query(Function).filter(Function.id == function_id).first()
    if db_function is None:
        raise HTTPException(status_code=404, detail="Function not found")
    
    # Delete code directory
    func_dir = os.path.join(CODE_DIR, str(function_id))
    if os.path.exists(func_dir):
        shutil.rmtree(func_dir)
    
    # Delete function from database
    db.delete(db_function)
    db.commit()
    return {"detail": f"Function '{db_function.name}' deleted successfully"}

# Endpoint to execute a function
@app.post("/functions/{function_id}/execute")
def execute_function(function_id: Optional[int] = None, db: Session = Depends(get_db)):
    if function_id is None:
        raise HTTPException(status_code=400, detail="Function ID must be provided")
    
    db_function = db.query(Function).filter(Function.id == function_id).first()
    if db_function is None:
        raise HTTPException(status_code=404, detail="Function not found")
    
    # Build Docker image
    image_tag = build_docker_image(function_id, db_function.name, db_function.language)
    
    # Run function in Docker
    logs = run_function_in_docker(image_tag, db_function.timeout)
    return {"logs": logs}

@app.post("/functions/execute")
def execute_multiple_functions(request: ExecuteMultipleFunctionsRequest, db: Session = Depends(get_db)):
    results = []
    for function_id in request.function_ids:
        db_function = db.query(Function).filter(Function.id == function_id).first()
        if db_function is None:
            results.append({"id": function_id, "error": "Function not found"})
            continue
        
        try:
            # Build Docker image
            image_tag = build_docker_image(function_id, db_function.name, db_function.language)
            
            # Run function in Docker
            logs = run_function_in_docker(image_tag, db_function.timeout)
            results.append({"id": function_id, "logs": logs})
        except HTTPException as e:
            results.append({"id": function_id, "error": e.detail})
    
    return {"results": results}