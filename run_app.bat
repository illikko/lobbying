@echo off
REM Activer l'environnement Conda "base"
CALL C:\Users\casta\anaconda3\Scripts\activate.bat base

REM Aller dans le dossier contenant ton app
cd /d "C:\Users\casta\Documents\TI\GroupesDeTravail\Lobbying\repo\"

REM Lancer l'application Streamlit
streamlit run app_serving.py

pause