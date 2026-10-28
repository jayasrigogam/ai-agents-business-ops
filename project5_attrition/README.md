# Project 5 - Employee Attrition Root-Cause Intelligence

## Files to upload to GitHub Codespaces
1. generate_data.py
2. attrition_analysis.py
3. requirements.txt
(hr_data.csv is created by step 1 - or upload it if you want the same data.)

## Run in the Codespaces terminal
pip install -r requirements.txt
python generate_data.py
python attrition_analysis.py

## Screenshots to take for submission
1. Terminal after `python generate_data.py` (dataset created)
2. Terminal output sections 1-3 (overview, department rates, model accuracy)
3. Terminal output sections 4-6 (root causes, risk list, recommendations)
4. Explorer panel showing the `outputs/` folder
5. Open each PNG in outputs/ (click the file in the Explorer) - 6 charts
6. outputs/retention_report.txt

## Optional LLM step (extra marks - "AI system")
export ANTHROPIC_API_KEY="your-key"
pip install anthropic
python attrition_analysis.py
