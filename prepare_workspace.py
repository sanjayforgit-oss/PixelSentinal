"""
PixelSentinel Workspace Initializer

Creates an isolated preprocessing workspace.

Run:

python scripts/setup_workspace.py
"""

from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent

WORKSPACE = PROJECT_ROOT / "pipeline_workspace"

FOLDERS = [

    "incoming",

    "raw",

    "processed",

    "metadata",

    "reports",

    "manifest",

    "train",

    "val",

    "test",

]


def create_directories():

    print("=" * 70)
    print("Creating Pipeline Workspace")
    print("=" * 70)

    WORKSPACE.mkdir(exist_ok=True)

    for folder in FOLDERS:

        directory = WORKSPACE / folder

        directory.mkdir(parents=True, exist_ok=True)

        print(f"[OK] {directory}")


def clean_workspace():

    print()
    print("=" * 70)
    print("Cleaning Previous Workspace")
    print("=" * 70)

    for folder in FOLDERS:

        directory = WORKSPACE / folder

        if not directory.exists():
            continue

        for item in directory.iterdir():

            if item.is_file():

                item.unlink()

            else:

                shutil.rmtree(item)

    print("Workspace cleaned.")


def main():

    create_directories()

    answer = input(
        "\nClean existing workspace? (y/n): "
    ).strip().lower()

    if answer == "y":

        clean_workspace()

    print()
    print("=" * 70)
    print("Workspace Ready")
    print("=" * 70)

    print()

    print("Copy your new downloaded GeoTIFF files into:")

    print()

    print(WORKSPACE / "incoming")

    print()

    print("Then run:")

    print()

    print("python -m preprocessing.preprocessing_pipeline")


if __name__ == "__main__":

    main()