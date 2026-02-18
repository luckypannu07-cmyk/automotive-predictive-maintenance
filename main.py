from src.load_data import load_training_data
from src.build_features import add_rul
from src.training.train_pipeline import run_training

DATA_PATH = "data/train_FD001.txt"

def main():
    df = load_training_data(DATA_PATH)
    df = add_rul(df)
    run_training(df)

if __name__ == "__main__":
    main()
