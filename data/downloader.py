import os
import subprocess
import logging

from utils.logger import setup_logger

logger = setup_logger("Downloader")

REPO_URL = "https://github.com/kevserbusrayildirim/FSMTSAD.git"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FSMTSAD")

def download_dataset():
    """
    Downloads the FSMTSAD dataset by cloning the GitHub repository.
    """
    if os.path.exists(DATA_DIR):
        logger.info(f"Dataset directory {DATA_DIR} already exists. Skipping download.")
        return

    logger.info(f"Cloning repository from {REPO_URL} into {DATA_DIR}...")
    try:
        subprocess.run(["git", "clone", REPO_URL, DATA_DIR], check=True)
        logger.info("Repository cloned successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to clone repository: {e}")
        raise

if __name__ == "__main__":
    download_dataset()
