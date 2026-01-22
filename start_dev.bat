@echo off
echo Starting Scriptura Development Servers...

REM Start Audio Server in new window
start "Audio Server" cmd /k "conda activate audio && python audio_app.py"

REM Start Video Server in new window
start "Video Server" cmd /k "conda activate video && python video_app.py"

REM Small delay to let dependency servers start first
timeout /t 3 /nobreak > nul

REM Start Main App in new window
start "Main App" cmd /k "python app.py"

echo All servers starting in separate windows.
