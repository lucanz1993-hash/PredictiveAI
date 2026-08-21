@echo off
"C:\Users\Luca Bovolenta\Desktop\Pers\Luca\ClaudeCode\quant-ranking\.venv\Scripts\python.exe" "C:\Users\Luca Bovolenta\Desktop\Pers\Luca\ClaudeCode\quant-ranking\src\company.py" >> "C:\Users\Luca Bovolenta\Desktop\Pers\Luca\ClaudeCode\quant-ranking\company\task_run.log" 2>&1
set PY_EXIT=%errorlevel%

rem Best-effort: pubblica ledger/reports/alerts su GitHub cosi' la routine
rem cloud del report settimanale li possa leggere (non ha accesso al PC
rem locale). Mai un git add -A: solo questi 4 file espliciti, per non
rem rischiare di pubblicare per errore altre cartelle del repo (es. P1-2024 IOM).
cd /d "C:\Users\Luca Bovolenta\Desktop\Pers\Luca\ClaudeCode"
if exist "quant-ranking\company\ledger.json" git add quant-ranking/company/ledger.json >> "quant-ranking\company\task_run.log" 2>&1
if exist "quant-ranking\company\reports.log" git add quant-ranking/company/reports.log >> "quant-ranking\company\task_run.log" 2>&1
if exist "quant-ranking\company\alerts.log" git add quant-ranking/company/alerts.log >> "quant-ranking\company\task_run.log" 2>&1
if exist "quant-ranking\company\known_gaps.json" git add quant-ranking/company/known_gaps.json >> "quant-ranking\company\task_run.log" 2>&1

git diff --cached --quiet
if errorlevel 1 (
    git commit -m "Auto-update company ledger/reports (daily run)" >> "quant-ranking\company\task_run.log" 2>&1
    git push origin master >> "quant-ranking\company\task_run.log" 2>&1
)

exit /b %PY_EXIT%
