import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.core.config import load_config
from src.rag.index import build_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rag.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    rag_cfg = config["rag"]
    build_index(
        corpus_path=rag_cfg["corpus_path"],
        embedding_model_path=config["models"]["embedding_model_path"],
        index_dir=rag_cfg["index_dir"],
        normalize=rag_cfg.get("dense", {}).get("normalize", True),
    )


if __name__ == "__main__":
    main()
