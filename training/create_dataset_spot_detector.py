import argparse
from pathlib import Path

import pandas as pd
from loguru import logger


def create_dataset_spot_detector(
    folder_label_one: str,
    folder_label_zero: str,
    output_path_csv: str,
):
    """
    output_csv: expecting pull_path "full/path/name.csv"
    """
    records = []

    label_map = {folder_label_one: 1, folder_label_zero: 0}

    for label_folder, label in label_map.items():
        label_folder = Path(label_folder)
        if not label_folder.exists():
            raise AssertionError(f"Folder does not exist: {label_folder}")

        for img_file in label_folder.glob("*.jpg"):
            records.append(
                {"image_path": str(img_file.resolve()), "label": label}  # absolute path
            )

    df = pd.DataFrame(records)
    output_path = Path(output_path_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(f"Dataset saved to: {output_path} ({len(df)} samples)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--one",
        dest="folder_label_one",
        required=True,
        type=str,
        help="Path to folder containing images of label 1",
    )

    parser.add_argument(
        "-z",
        "--zero",
        dest="folder_label_zero",
        required=True,
        type=str,
        help="Path to folder containing images of label 0",
    )
    parser.add_argument(
        "-p",
        "--output-path",
        dest="output_path_csv",
        required=True,
        type=str,
        help="Path to output CSV file",
    )

    args = parser.parse_args()
    create_dataset_spot_detector(
        folder_label_one=args.folder_label_one,
        folder_label_zero=args.folder_label_zero,
        output_path_csv=args.output_path_csv,
    )
