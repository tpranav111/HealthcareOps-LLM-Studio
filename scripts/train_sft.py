import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.core.config import load_config
from src.training.sft import train_sft


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sft.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    train_sft(config)


if __name__ == "__main__":
    main()
