@echo off
REM Aquaculture Data Sync Batch File
REM This batch file runs the main.py script with headless mode
REM Ensure Python is in your PATH or update the path below

REM Change to the script directory (equivalent to "Start in" in Task Scheduler)
cd /d "C:\Users\Ang Wei Ding\Desktop\FYP\dashboard_design"

REM Run the Python script with headless flag
.\.venv\Scripts\python.exe main.py --headless

REM Optional: Add a pause if you want to see output (remove for silent operation)
REM pause