import yaml
import pandas as pd
from pathlib import Path

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_folder(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def read_fundamentals(path):
    df = pd.read_csv(path)
    
    # normalize ticker column
    df['ticker'] = df['ticker'].astype(str)
    return df.set_index('ticker').to_dict(orient='index')
