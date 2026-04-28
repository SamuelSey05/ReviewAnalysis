import os 
import urllib.request
import logging

logger = logging.getLogger(__name__)

aware_url = "https://zenodo.org/records/5528481/files/AWARE_Comprehensive.csv"

def verify_aware_dataset_is_present(dataset_path: str):
    """Downloads the AWARE dataset to datsets/AWARE_Comprehensive.csv if it is not already present

    Args:
        dataset_path (str): Path where the AWARE dataset should be located
    """

    if os.path.exists(dataset_path):
        logger.info(f"AWARE dataset already exists at {dataset_path}. Skipping download.")
        return
    
    logger.info("Downloading AWARE dataset...")

    try:
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        urllib.request.urlretrieve(aware_url, dataset_path)
        logger.info(f"Successfully downloaded AWARE dataset to {dataset_path}")
    except Exception as e:
        logger.error(f"Failed to download AWARE dataset: {e}")
        raise RuntimeError(f"Failed to download AWARE dataset: {e}")
