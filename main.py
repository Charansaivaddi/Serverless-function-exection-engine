import time
import os
import shutil
import subprocess
import docker
import signal
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from db import Base, engine, SessionLocal, Function, FunctionMetrics
from fastapi.middleware.cors import CORSMiddleware
from docker.errors import DockerException
from collections import defaultdict
from threading import Lock
import tarfile
import io, threading
from fastapi.requests import Request
import resource
import datetime
from sqlalchemy import func
from datetime import datetime, timedelta

# Try to import psutil, with fallback for measuring resources
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil module not found. Falling back to basic resource measurement.")

global_container_pool = []
POOL_SIZE = 5
POOL_LOCK = Lock()

# In-memory container pool: maps image_tag → list of WarmContainer objects
warm_container_pool = defaultdict(list)

# How long to keep containers warm (in seconds)
WARM_EXPIRY_SECONDS = 300  # 5 minutes

class WarmContainer:
    def __init__(self, container, timestamp):
        self.container = container
        self.timestamp = timestamp

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


def run_with_timeout(container, cmd, timeout):
    result = {}

    def target():
        try:
            # Start measuring time
            start_time = time.time()
            
            # Measure resource usage at start
            if HAS_PSUTIL:
                process = psutil.Process(os.getpid())
                start_cpu_times = process.cpu_times()
                start_memory = process.memory_info().rss / 1024 / 1024  # MB
            else:
                # Fallback to resource module for basic measurements
                start_usage = resource.getrusage(resource.RUSAGE_SELF)
            
            # Execute the command in the container
            exec_result = container.exec_run(cmd)
            result['output'] = exec_result.output.decode("utf-8") if exec_result.output else ""
            result['exit_code'] = exec_result.exit_code
            
            # Measure resource usage at end
            end_time = time.time()
            if HAS_PSUTIL:
                end_cpu_times = process.cpu_times()
                end_memory = process.memory_info().rss / 1024 / 1024  # MB
                
                # Calculate user + system CPU time in milliseconds
                cpu_user_time = (end_cpu_times.user - start_cpu_times.user) * 1000
                cpu_system_time = (end_cpu_times.system - start_cpu_times.system) * 1000
                total_cpu_time = cpu_user_time + cpu_system_time
                memory_used = end_memory - start_memory if end_memory > start_memory else 0.1
                peak_memory = max(end_memory, start_memory)
            else:
                # Fallback for systems without psutil
                end_usage = resource.getrusage(resource.RUSAGE_SELF)
                cpu_user_time = (end_usage.ru_utime - start_usage.ru_utime) * 1000
                cpu_system_time = (end_usage.ru_stime - start_usage.ru_stime) * 1000
                total_cpu_time = cpu_user_time + cpu_system_time
                memory_used = end_usage.ru_maxrss / 1024  # Convert to MB
                peak_memory = memory_used
            
            # Calculate execution time
            execution_time = (end_time - start_time) * 1000  # convert to ms
            
            # Store detailed metrics
            result['metrics'] = {
                'cpu_time': total_cpu_time,
                'cpu_user_time': cpu_user_time,
                'cpu_system_time': cpu_system_time,
                'memory_used': memory_used,
                'exec_time': execution_time,
                'peak_memory': peak_memory
            }
            
            print(f"CPU Time: {total_cpu_time:.2f}ms (User: {cpu_user_time:.2f}ms, System: {cpu_system_time:.2f}ms)")
            print(f"Memory Used: {memory_used:.2f} MB (Peak: {peak_memory:.2f} MB)")
            print(f"Execution Time: {execution_time:.2f} ms")
            
        except Exception as e:
            result['error'] = str(e)
            print(f"Error in execution thread: {str(e)}")

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        container.kill()
        raise HTTPException(status_code=408, detail="Function execution timed out")

    if 'error' in result:
        raise HTTPException(status_code=500, detail=f"Execution error: {result['error']}")

    return result

      
def make_tar_archive(src_path, dest_filename):
    """Create a tar archive containing the function file (for Docker copy)."""
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w") as tar:
        tar.add(src_path, arcname=dest_filename)
    data.seek(0)
    return data

def run_function_in_docker(code_path, language, timeout, runtime):
    """Run a function using a pre-warmed shared Docker container based on runtime and language."""
    if not docker_client:
        raise HTTPException(status_code=500, detail="Docker is not available.")

    with POOL_LOCK:
        # Ensure pool exists for the given language and runtime
        if language not in WARM_CONTAINER_POOLS or runtime not in WARM_CONTAINER_POOLS[language]:
            raise HTTPException(status_code=400, detail=f"Unsupported language or runtime: {language}, {runtime}")

        # Get container pool
        if not WARM_CONTAINER_POOLS[language][runtime]:
            raise HTTPException(status_code=503, detail=f"No available {language} containers for runtime {runtime}.")

        warm = WARM_CONTAINER_POOLS[language][runtime].pop(0)
        print(f"🚀 Assigned {language} container [{runtime}]: {warm.container.id[:12]} to run the function.")

    try:
        # Determine destination filename
        ext = ".py" if language == "python" else ".js"
        dest_filename = f"main{ext}"

        # Copy code into container
        archive = make_tar_archive(code_path, dest_filename)
        warm.container.put_archive("/mnt", archive)

        # Build command
        exec_cmd = f"python3 /mnt/{dest_filename}" if language == "python" else f"node /mnt/{dest_filename}"

        # Execute with timeout
        result = run_with_timeout(warm.container, exec_cmd, timeout)
        output = result.get('output', '')
        exit_code = result.get('exit_code', 1)
        metrics = result.get('metrics', {})

        # Re-add container to pool
        with POOL_LOCK:
            WARM_CONTAINER_POOLS[language][runtime].append(WarmContainer(warm.container, time.time()))

        if exit_code != 0:
            error_msg = f"Function exited with code {exit_code}: {output}"
            print(error_msg)
            return error_msg, metrics

        return output, metrics

    except Exception as e:
        error_msg = f"Error running code in warm container: {str(e)}"
        print(error_msg)
        try:
            warm.container.remove(force=True)
        except Exception:
            pass  # Already removed or otherwise inaccessible
        
        return error_msg, {}


def store_metrics(db: Session, function_id: int, metrics: dict, runtime: str):
    """Store execution metrics in the database."""
    db_metrics = FunctionMetrics(
        function_id=function_id,
        execution_time=metrics['execution_time'],
        cpu_time=metrics['cpu_time'],
        memory_used=metrics['memory_used'],
        exit_code=metrics['exit_code'],
        runtime=runtime,
        success=metrics['exit_code'] == 0
    )
    db.add(db_metrics)
    db.commit()
    return db_metrics


def get_aggregated_metrics(db: Session, function_id: int, time_window: str = "24h"):
    """Get aggregated metrics for a function."""
    now = datetime.utcnow()

    if time_window == "1h":
        delta = timedelta(hours=1)
    elif time_window == "24h":
        delta = timedelta(days=1)
    elif time_window == "7d":
        delta = timedelta(days=7)
    else:
        delta = timedelta(days=1)  # default

    # Query metrics for the time window
    metrics = db.query(FunctionMetrics).filter(
        FunctionMetrics.function_id == function_id,
        FunctionMetrics.timestamp >= now - delta
    ).all()

    if not metrics:
        return {
            "function_id": function_id,
            "time_window": time_window,
            "count": 0,
            "success_rate": 0,
            "avg_execution_time": 0,
            "max_execution_time": 0,
            "min_execution_time": 0,
            "avg_cpu_time": 0,
            "avg_memory_used": 0,
            "max_memory_used": 0,
            "successes": 0,
            "failures": 0
        }

    # Calculate aggregates
    execution_times = [m.execution_time for m in metrics]
    cpu_times = [m.cpu_time for m in metrics]
    memory_used = [m.memory_used for m in metrics]
    successes = sum(1 for m in metrics if m.success)
    failures = len(metrics) - successes

    return {
        "function_id": function_id,
        "time_window": time_window,
        "count": len(metrics),
        "success_rate": successes / len(metrics) if metrics else 0,
        "avg_execution_time": sum(execution_times) / len(execution_times) if execution_times else 0,
        "max_execution_time": max(execution_times) if execution_times else 0,
        "min_execution_time": min(execution_times) if execution_times else 0,
        "avg_cpu_time": sum(cpu_times) / len(cpu_times) if cpu_times else 0,
        "avg_memory_used": sum(memory_used) / len(memory_used) if memory_used else 0,
        "max_memory_used": max(memory_used) if memory_used else 0,
        "successes": successes,
        "failures": failures
    }


@app.on_event("startup")
def initialize_global_container_pool():
    base_images = {
        "python": "python:3.10-slim",
        "javascript": "node:18-slim"
    }

    runtimes = ["runc", "runsc"]
    containers_per_combination = 2  # 2 per language per runtime
    now = time.time()

    global WARM_CONTAINER_POOLS
    WARM_CONTAINER_POOLS = {
        lang: {runtime: [] for runtime in runtimes}
        for lang in base_images
    }

    for lang, image in base_images.items():
        for runtime in runtimes:
            for i in range(containers_per_combination):
                try:
                    container = docker_client.containers.run(
                        image,
                        detach=True,
                        tty=True,
                        command="tail -f /dev/null",
                        runtime=runtime
                    )
                    WARM_CONTAINER_POOLS[lang][runtime].append(WarmContainer(container, now))
                    print(f"✅ {lang.capitalize()} container [{runtime}] {i+1} started: {container.id[:12]}")
                except Exception as e:
                    print(f"❌ Failed to start {lang} container [{runtime}] {i+1}: {e}")



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
@app.post("/functions/{function_id}/execute_docker")
async def execute_function_docker(request: Request, function_id: int, db: Session = Depends(get_db)):
    db_function = db.query(Function).filter(Function.id == function_id).first()
    if db_function is None:
        raise HTTPException(status_code=404, detail="Function not found")
    
    try:
        # Build Docker image
        image_tag = build_docker_image(function_id, db_function.name, db_function.language)
        
        # Run function in Docker
        logs, metrics = run_function_in_docker(db_function.code_path, db_function.language, db_function.timeout, 'runc')
        
        # Save metrics to database if they exist
        if isinstance(metrics, dict) and metrics:
            try:
                save_execution_metrics(db, function_id, metrics, 'runc')
            except Exception as e:
                print(f"Error saving metrics: {str(e)}")
    
        return {"logs": logs, "metrics": metrics}
    except Exception as e:
        print(f"Error in execute_function_docker: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
# Endpoint to execute a function with GVisor
@app.post("/functions/{function_id}/execute_gvisor")
async def execute_function_gvisor(request: Request, function_id: int, db: Session = Depends(get_db)):
    db_function = db.query(Function).filter(Function.id == function_id).first()
    if db_function is None:
        raise HTTPException(status_code=404, detail="Function not found")
    
    try:
        # Build Docker image
        image_tag = build_docker_image(function_id, db_function.name, db_function.language)
        
        # Run function in Docker with GVisor
        logs, metrics = run_function_in_docker(db_function.code_path, db_function.language, db_function.timeout, 'runsc')
        
        # Save metrics to database if they exist
        if isinstance(metrics, dict) and metrics:
            try:
                save_execution_metrics(db, function_id, metrics, 'runsc')
            except Exception as e:
                print(f"Error saving metrics: {str(e)}")
    
        return {"logs": logs, "metrics": metrics}
    except Exception as e:
        print(f"Error in execute_function_gvisor: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/functions/{function_id}/metrics")
def get_function_metrics(
        function_id: int,
        time_window: str = "24h",
        db: Session = Depends(get_db)
):
    """Get aggregated metrics for a function."""
    db_function = db.query(Function).filter(Function.id == function_id).first()
    if db_function is None:
        raise HTTPException(status_code=404, detail="Function not found")

    metrics = get_aggregated_metrics(db, function_id, time_window)
    if not metrics:
        raise HTTPException(status_code=404, detail="No metrics found for this function")

    return metrics


@app.get("/functions/{function_id}/metrics/raw")
def get_raw_metrics(
        function_id: int,
        limit: int = 100,
        db: Session = Depends(get_db)
):
    """Get raw metrics data for a function."""
    db_function = db.query(Function).filter(Function.id == function_id).first()
    if db_function is None:
        raise HTTPException(status_code=404, detail="Function not found")

    metrics = db.query(FunctionMetrics).filter(
        FunctionMetrics.function_id == function_id
    ).order_by(
        FunctionMetrics.timestamp.desc()
    ).limit(limit).all()

    return metrics

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
            logs, metrics = run_function_in_docker(db_function.code_path, db_function.language, db_function.timeout, 'runc')
            
            # Save metrics to database if they exist
            if isinstance(metrics, dict) and metrics:
                try:
                    save_execution_metrics(db, function_id, metrics, 'runc')
                except Exception as e:
                    print(f"Error saving metrics for function {function_id}: {str(e)}")

            results.append({"id": function_id, "logs": logs, "metrics": metrics})
        except HTTPException as e:
            results.append({"id": function_id, "error": e.detail})
        except Exception as e:
            results.append({"id": function_id, "error": str(e)})
    
    return {"results": results}

def save_execution_metrics(db: Session, function_id: int, metrics: dict, runtime: str):
    """Store execution metrics in the database."""
    try:
        # Adapt metrics to match FunctionMetrics model
        db_metrics = FunctionMetrics(
            function_id=function_id,
            execution_time=metrics.get('exec_time', 0) / 1000,  # Convert ms to seconds
            cpu_time=metrics.get('cpu_time', 0) / 1000,  # Convert ms to seconds
            memory_used=int(metrics.get('memory_used', 0) * 1024),  # Convert MB to KB
            exit_code=0,  # Assume success
            runtime=runtime,
            success=True
        )
        db.add(db_metrics)
        db.commit()
        return db_metrics
    except Exception as e:
        print(f"Error saving metrics to database: {str(e)}")
        db.rollback()
        return None

@app.get("/metrics/function/{function_id}")
def get_function_metrics(function_id: int, db: Session = Depends(get_db)):
    """Get metrics for a specific function."""
    db_function = db.query(Function).filter(Function.id == function_id).first()
    if db_function is None:
        raise HTTPException(status_code=404, detail="Function not found")
    
    # Query for metrics
    metrics = db.query(
        func.avg(FunctionMetrics.cpu_time).label("avg_cpu_time"),
        func.avg(FunctionMetrics.memory_used).label("avg_memory"),
        func.avg(FunctionMetrics.execution_time).label("avg_exec_time"),
        func.count(FunctionMetrics.id).label("total_executions")
    ).filter(FunctionMetrics.function_id == function_id).first()
    
    # Get recent history
    history = db.query(FunctionMetrics).filter(
        FunctionMetrics.function_id == function_id
    ).order_by(FunctionMetrics.timestamp.desc()).limit(10).all()
    
    # Convert metrics to response format
    response = {
        "function_name": db_function.name,
        "avg_cpu_time": float(metrics.avg_cpu_time * 1000 if metrics.avg_cpu_time else 0),  # Convert to ms
        "avg_memory": float(metrics.avg_memory / 1024 if metrics.avg_memory else 0),  # Convert to MB
        "avg_exec_time": float(metrics.avg_exec_time * 1000 if metrics.avg_exec_time else 0),  # Convert to ms
        "total_executions": metrics.total_executions or 0,
        "history": []
    }
    
    # Add history data
    for metric in history:
        response["history"].append({
            "timestamp": metric.timestamp.isoformat(),
            "cpu_time": float(metric.cpu_time * 1000),  # Convert to ms
            "memory_used": float(metric.memory_used / 1024),  # Convert to MB
            "exec_time": float(metric.execution_time * 1000),  # Convert to ms
            "runtime": metric.runtime,
        })
    
    return response

@app.get("/metrics/summary")
def get_metrics_summary(db: Session = Depends(get_db)):
    """Get summary metrics for all functions"""

    # Overall stats
    overall_stats = db.query(
        func.count(FunctionMetrics.id).label("total_executions"),
        func.avg(FunctionMetrics.cpu_time).label("avg_cpu_time"),
        func.avg(FunctionMetrics.memory_used).label("avg_memory"),
        func.avg(FunctionMetrics.execution_time).label("avg_exec_time")
    ).first()

    # Per-runtime stats
    runtime_stats = db.query(
        FunctionMetrics.runtime,
        func.avg(FunctionMetrics.cpu_time).label("avg_cpu_time"),
        func.avg(FunctionMetrics.memory_used).label("avg_memory"),
        func.avg(FunctionMetrics.execution_time).label("avg_exec_time"),
        func.count(FunctionMetrics.id).label("count")
    ).group_by(FunctionMetrics.runtime).all()

    # Per-function stats (use explicit join condition)
    function_stats = db.query(
        Function.id,
        Function.name,
        func.avg(FunctionMetrics.cpu_time).label("avg_cpu_time"),
        func.avg(FunctionMetrics.memory_used).label("avg_memory"),
        func.avg(FunctionMetrics.execution_time).label("avg_exec_time"),
        func.count(FunctionMetrics.id).label("total_executions")
    ).join(FunctionMetrics, Function.id == FunctionMetrics.function_id).group_by(Function.id, Function.name).all()

    # Recent executions for timeline (use explicit join condition)
    recent_executions = db.query(
        FunctionMetrics.id,
        FunctionMetrics.function_id,
        Function.name.label("function_name"),
        FunctionMetrics.cpu_time,
        FunctionMetrics.memory_used,
        FunctionMetrics.execution_time,
        FunctionMetrics.runtime,
        FunctionMetrics.timestamp
    ).join(Function, Function.id == FunctionMetrics.function_id).order_by(FunctionMetrics.timestamp.desc()).limit(50).all()

    execution_history = []
    for execution in recent_executions:
        execution_history.append({
            "id": execution.id,
            "function_id": execution.function_id,
            "function_name": execution.function_name,
            "cpu_time": execution.cpu_time * 1000 if execution.cpu_time else 0,
            "memory_used": execution.memory_used / 1024 if execution.memory_used else 0,
            "exec_time": execution.execution_time * 1000 if execution.execution_time else 0,
            "runtime": execution.runtime,
            "timestamp": execution.timestamp.isoformat() if execution.timestamp else None
        })

    function_metrics = []
    for stat in function_stats:
        function_metrics.append({
            "function_id": stat.id,
            "function_name": stat.name,
            "avg_cpu_time": stat.avg_cpu_time * 1000 if stat.avg_cpu_time else 0,
            "avg_memory": stat.avg_memory / 1024 if stat.avg_memory else 0,
            "avg_exec_time": stat.avg_exec_time * 1000 if stat.avg_exec_time else 0,
            "total_executions": stat.total_executions or 0
        })

    runtime_metrics = []
    for stat in runtime_stats:
        runtime_metrics.append({
            "runtime": stat.runtime,
            "avg_cpu_time": stat.avg_cpu_time * 1000 if stat.avg_cpu_time else 0,
            "avg_memory": stat.avg_memory / 1024 if stat.avg_memory else 0,
            "avg_exec_time": stat.avg_exec_time * 1000 if stat.avg_exec_time else 0,
            "total_executions": stat.count or 0
        })

    return {
        "total_executions": overall_stats.total_executions or 0,
        "avg_cpu_time": overall_stats.avg_cpu_time * 1000 if overall_stats.avg_cpu_time else 0,
        "avg_memory": overall_stats.avg_memory / 1024 if overall_stats.avg_memory else 0,
        "avg_exec_time": overall_stats.avg_exec_time * 1000 if overall_stats.avg_exec_time else 0,
        "execution_history": execution_history,
        "function_metrics": function_metrics,
        "runtime_metrics": runtime_metrics
    }

@app.get("/metrics/logs")
def get_logs_metrics(db: Session = Depends(get_db)):
    """
    Return number of executions per hour.
    """
    rows = (
        db.query(
            func.strftime('%Y-%m-%dT%H:00:00', FunctionMetrics.timestamp).label('time'),
            func.count(FunctionMetrics.id).label('count')
        )
        .group_by('time')
        .order_by('time')
        .all()
    )
    return [{"timestamp": r.time, "count": r.count} for r in rows]
