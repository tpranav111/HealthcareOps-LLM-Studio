import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import uvicorn

from src.core.config import deep_merge, load_config
from src.serving.app import create_app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/serving.yaml")
    parser.add_argument("--rag-config", default="configs/rag.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    rag_config = load_config(args.rag_config)
    config = deep_merge(config, {"rag": rag_config.get("rag", {})})

    app = create_app(config)
    uvicorn.run(app, host=config["serving"]["host"], port=config["serving"]["port"])


if __name__ == "__main__":
    main()
