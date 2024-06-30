#!/usr/bin/env python
import argparse
import logging
import os

import numpy as np
import pandas as pd
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# HACK: see https://stackoverflow.com/questions/74844262/how-can-i-solve-error-module-numpy-has-no-attribute-float-in-python
np.float = float


def go(args):

    run = wandb.init(project="exercise_5", job_type="process_data")

    ## YOUR CODE HERE
    logger.info("Get data and start preprocessing")
    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()

    df = pd.read_parquet(artifact_path)

    df.drop_duplicates().reset_index(drop=True)

    # NOTE: again, in a real setting, you will have to make sure that your feature
    # store provides this text_feature at inference time, OR, you will have to move
    # the computation of this feature to the inference pipeline.
    df["title"].fillna(value="", inplace=True)
    df["song_name"].fillna(value="", inplace=True)
    df["text_feature"] = df["title"] + " " + df["song_name"]

    logger.info("Create new artifact from preprocessed data")
    file_name = "preprocessed_data.csv"
    df.to_csv(file_name)

    preprocessed_artifact = wandb.Artifact(
        name=args.artifact_name,
        type=args.artifact_type,
        description=args.artifact_description,
    )
    preprocessed_artifact.add_file(file_name)

    logger.info("Upload/Log perprocessed artifact")
    run.log_artifact(preprocessed_artifact)
    os.remove(file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess a dataset",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True,
    )

    parser.add_argument(
        "--artifact_name",
        type=str,
        help="Name for the artifact",
        required=True,
    )

    parser.add_argument(
        "--artifact_type",
        type=str,
        help="Type for the artifact",
        required=True,
    )

    parser.add_argument(
        "--artifact_description",
        type=str,
        help="Description for the artifact",
        required=True,
    )

    args = parser.parse_args()

    go(args)
