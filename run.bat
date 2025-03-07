@echo off
REM Create a virtual environment if it does not exist
if not exist "venv" (
    python -m venv venv
)

REM Activate the virtual environment
call venv\Scripts\activate

REM Install packages
pip install -r requirements.txt

REM Run the FastAPI server
uvicorn app.main:app --reload

pause