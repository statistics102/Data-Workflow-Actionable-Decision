@echo off
echo Installing required packages...
pip install -r requirements.txt

echo Generating sample data...
python generate_sample_data.py

echo Running complete analysis pipeline...
python complete_analysis_pipeline.py

echo Analysis completed! Check the generated files.
pause