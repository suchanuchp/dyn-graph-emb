import argparse
from nilearn import datasets
# python fetch_abide.py ../data


def fetch_abide_data(data_dir):
    # Fetching the ABIDE dataset with the specified parameters
    abide_data = datasets.fetch_abide_pcp(data_dir=data_dir, pipeline="cpac", derivatives=["func_preproc"])
    return abide_data


def main():
    parser = argparse.ArgumentParser(description="Fetch ABIDE dataset with user-specified data directory.")
    parser.add_argument("data_dir", type=str, help="Directory to store the fetched data")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Fetch the data
    data = fetch_abide_data(args.data_dir)

    # Print some basic information about the fetched data as an example
    print("Data keys available:", data.keys())
    print("Number of subjects fetched:", len(data.func_preproc))


if __name__ == "__main__":
    main()
