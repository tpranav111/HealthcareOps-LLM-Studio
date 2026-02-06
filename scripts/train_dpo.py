import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.core.config import load_config
from src.training.dpo import train_dpo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dpo.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    train_dpo(config)


if __name__ == "__main__":
    main()
