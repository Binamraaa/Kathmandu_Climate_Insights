@echo off
REM K Laune: run the evening forecast (scheduled daily at 20:00 via Task Scheduler)
cd /d C:\Users\ALiEN\Kathmandu_climate_history
call C:\Users\ALiEN\anaconda3\Scripts\activate.bat ktm-climate
if not exist data\verification mkdir data\verification
echo ==== %date% %time% ==== >> data\verification\export_log.txt
python -m src.export_forecast >> data\verification\export_log.txt 2>&1
