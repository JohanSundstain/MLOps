"""Single entry point of the project.

Usage::

    python commands.py --help
    python commands.py download
    python commands.py preprocess
    python commands.py train --overrides="train.max_epochs=30,train.batch_size=16"

Every command composes the Hydra config from ``configs/`` (see ``lake_segmentation.cli``).
"""

from lake_segmentation.cli import main

if __name__ == "__main__":
    main()
