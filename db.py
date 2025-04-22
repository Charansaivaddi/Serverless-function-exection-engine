# db.py
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from datetime import datetime
from sqlalchemy import DateTime






DATABASE_URL = "sqlite:///./functions.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Function(Base):
    __tablename__ = "functions"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    route = Column(String)
    language = Column(String)
    timeout = Column(Float)
    code_path = Column(String)

class FunctionMetrics(Base):
    __tablename__ = "function_metrics"

    id = Column(Integer, primary_key=True, index=True)
    function_id = Column(Integer, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    execution_time = Column(Float)  # in seconds
    cpu_time = Column(Float)  # in seconds
    memory_used = Column(Integer)  # in KB
    exit_code = Column(Integer)
    runtime = Column(String)  # 'runc' or 'runsc'
    success = Column(Boolean)