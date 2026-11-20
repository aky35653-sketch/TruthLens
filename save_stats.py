
from pathlib import Path
import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "Dataset"
MODEL_DIR = BASE_DIR / "models"

true_df = pd.read_csv(DATASET_DIR / "True.csv")
fake_df = pd.read_csv(DATASET_DIR / "Fake.csv")

stats = {
    "total_articles": len(true_df) + len(fake_df),
    "real_articles": len(true_df),
    "fake_articles": len(fake_df)
}

joblib.dump(stats, MODEL_DIR / "dataset_stats.pkl", compress=3)

print("Dataset statistics saved successfully!")
print(stats)
