COVID-19 Anomaly Detection with an AI Agent

This repository demonstrates an end-to-end agentic anomaly detection and handling pipeline on live COVID-19 time-series data. 

load_live_covid_data
. Data is ingested from the public disease.sh API, then statistical methods (Z-score spike detection and growth-rate thresholds) are used to flag anomalies and classify severity (MINOR, WARNING, CRITICAL) 

anomaly_ai_agent
. A GroqCloud-powered AI Agent autonomously decides whether to fix, keep, or flag anomalies for human review 

detect_anomalies
. Minor anomalies are auto-corrected using rolling mean smoothing, and results are visualized through a Streamlit dashboard 
