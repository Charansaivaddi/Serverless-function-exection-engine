#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Recreate database with updated schema
rm -f functions.db
python3 -c "from db import Base, engine; Base.metadata.create_all(bind=engine)"

echo "Dependencies installed and database recreated!"
echo "Start the application with: uvicorn main:app --reload"
