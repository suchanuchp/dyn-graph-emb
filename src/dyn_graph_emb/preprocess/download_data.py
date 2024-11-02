import argparse
import logging
from nilearn import datasets


def setup_logging(log_file):
    """Set up the logging configuration."""
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def fetch_abide_data(data_dir):
    """Fetch ABIDE dataset with specified parameters."""
    logging.info("Starting to fetch ABIDE data.")
    abide_data = datasets.fetch_abide_pcp(data_dir=data_dir, pipeline="cpac", derivatives=["func_preproc"])
    logging.info("ABIDE data fetched successfully.")
    return abide_data


def main():
    parser = argparse.ArgumentParser(description="Fetch ABIDE dataset with user-specified data directory.")
    parser.add_argument("data_dir", type=str, help="Directory to store the fetched data")
    parser.add_argument("--log", type=str, default="abide_fetch.log", help="Path to the log file.")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log)

    # Fetch the data
    data = fetch_abide_data(args.data_dir)

    # Log some basic information about the fetched data
    logging.info(f"Data keys available: {list(data.keys())}")
    logging.info(f"Number of subjects fetched: {len(data.func_preproc)}")


if __name__ == "__main__":
    main()
