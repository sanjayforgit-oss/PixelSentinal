"""
PixelSentinel Dataset Merger
Safely copies newly generated dataset splits from the pipeline workspace 
into the main training directory without overwriting existing old data.
"""

import shutil
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("PixelSentinel.DatasetMerger")

def merge_datasets(source_dir: str, dest_dir: str) -> None:
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)

    if not source_path.exists():
        logger.error(f"Source directory '{source_path}' does not exist.")
        return

    splits = ["train", "val", "test"]
    subdirs = ["input", "target"]

    total_copied = 0
    total_skipped = 0

    for split in splits:
        for subdir in subdirs:
            src_folder = source_path / split / subdir
            dst_folder = dest_path / split / subdir

            if not src_folder.exists():
                logger.warning(f"Source folder missing, skipping: {src_folder}")
                continue

            # Ensure the destination folder exists (creates it if it's a fresh setup)
            dst_folder.mkdir(parents=True, exist_ok=True)

            # Find all tif/tiff files
            files_to_copy = list(src_folder.glob("*.tif")) + list(src_folder.glob("*.tiff"))
            
            for file_path in files_to_copy:
                dst_file = dst_folder / file_path.name
                
                # Safety Check: Do not overwrite old files
                if dst_file.exists():
                    total_skipped += 1
                else:
                    shutil.copy2(file_path, dst_file)
                    total_copied += 1

    logger.info("=" * 60)
    logger.info("DATASET MERGE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"New Files Copied   : {total_copied}")
    logger.info(f"Old Files Kept     : {total_skipped}")
    logger.info("=" * 60)

if __name__ == "__main__":
    # Define paths based on your project configuration
    PIPELINE_WORKSPACE = "pipeline_workspace"
    MAIN_DATASET = "datasets"
    
    logger.info(f"Merging data from '{PIPELINE_WORKSPACE}' into '{MAIN_DATASET}'...")
    merge_datasets(PIPELINE_WORKSPACE, MAIN_DATASET)