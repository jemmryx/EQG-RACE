import os
import sys
import time

import config
from infenrence import BeamSearcher
from trainer import Trainer


def main():
    mode = sys.argv[1]
    if mode == "train":
        if config.fine_tune:
            model_path = config.fine_tune_path
        else:
            model_path = None

        trainer = Trainer(model_path=model_path, level=config.level)
        trainer.train()
    else:
        beamsearcher = BeamSearcher(config.model_path, config.output_dir)
        beamsearcher.decode()


if __name__ == "__main__":
    main()
