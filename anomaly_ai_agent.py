from phi.agent import Agent
from phi.model.groq import Groq
from load_live_covid_data import load_live_covid_data
from detect_anomalies import *
from dotenv import load_dotenv
from tabulate import tabulate
import streamlit as st

# ---------------------------------------
# CONFIG
# ---------------------------------------
COUNTRY = "India"
DAYS = 150
VALID_ACTIONS = ["FIX_ANOMALY", "KEEP_ANOMALY", "FLAG_FOR_REVIEW"]

load_dotenv()
df = load_live_covid_data(COUNTRY,DAYS)
df = detect_anomalies(df)
df = compute_severity(df)
print("\nCOVID Data Table of " + COUNTRY + " with Severity")
print(tabulate(df , headers="keys", tablefmt="fancy_grid"))


# ---------------------------------------
# AGENT PROMPT
# ---------------------------------------
def build_agent_prompt(obs):
    return f"""
You are an AI monitoring agent for COVID-19 data.

Observed anomaly:
Date: {obs['date']}
Cases: {obs['cases']}
Severity: {obs['severity']}

Decision rules:
- FIX_ANOMALY: noise, reporting fluctuation
- KEEP_ANOMALY: real outbreak signal
- FLAG_FOR_REVIEW: severe or ambiguous anomaly

Respond with ONLY one of:
FIX_ANOMALY
KEEP_ANOMALY
FLAG_FOR_REVIEW
"""

# ---------------------------------------
# BUILDING AI AGENT
# ---------------------------------------
agent = Agent(
    name="CovidAnomalyAgent",
    model=Groq(id="openai/gpt-oss-120b"),
    instructions="""
You are an AI agent monitoring live COVID-19 time-series data.
Detect anomalies, decide according to the anomaly:
"FIX_ANOMALY", "KEEP_ANOMALY", "FLAG_FOR_REVIEW"."""
)
for i in range(len(df)):
    if df.loc[i, "Anomaly"] == "YES":
        obs = build_observation(df, i)
        prompt = build_agent_prompt(obs)
        response = agent.run(prompt)

        decision = response.messages[-1].content.strip()
        decision = decision if decision in VALID_ACTIONS else "FLAG_FOR_REVIEW"
        df = agent_action(df, i, decision)

# ---------------------------------------
# FINAL TABLE
# ---------------------------------------
final_table = df[[
        "Date",
        "Cases",
        "Anomaly",
        "Severity",
        "Agent Decision",
        "Action"
    ]]

#print(final_table.head(20))

#print("\nCOVID Analysis with AI Agent Decision and Action")
#print(tabulate(final_table , headers="keys", tablefmt="fancy_grid"))

st.title("COVID-19 Anomaly Detection Report")
def highlight_anomalies(row):
    if row["Anomaly"] == "YES":
        return ["background-color: #ffcccc"] * len(row)  # light red
    return [""] * len(row)

st.dataframe(
    final_table.style.apply(highlight_anomalies, axis=1),
    use_container_width=True
)