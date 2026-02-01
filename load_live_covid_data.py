import requests
import pandas as pd

# ---------------------------------------
# DATA INGESTION (disease.sh)
# ---------------------------------------

def load_live_covid_data(country: str , days:int):
    url = f"https://disease.sh/v3/covid-19/historical/{country}?lastdays={days}"
    response = requests.get(url)
    data = response.json()["timeline"]["cases"]

    df = (
        pd.DataFrame(list(data.items()), columns=["Date", "Cases"])
        .assign(Date=lambda d: pd.to_datetime(d["Date"], format="%m/%d/%y"))
        .sort_values("Date")
        .reset_index(drop=True)
    )
    #print(df)
    return df

#load_live_covid_data("india",100)