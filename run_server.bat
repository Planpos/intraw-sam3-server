@echo off
cd /d C:\Users\intraw_sewon\Desktop\workspace\intraw-sam3-server
set SAM_API_KEY=F8ibaj2yoLqyYUp7sOZofcGflrHcWTox8DDMYYiAd7qwzPTWjAp304p5z2lci
C:\Users\intraw_sewon\miniconda3\envs\sam3\python.exe -m uvicorn server:app --host 0.0.0.0 --port 8001
pause
