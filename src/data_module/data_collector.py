"""Collect data from source API."""
import json
import re
import time
from pathlib import Path
from typing import Union

import pandas as pd
import requests


def get_query(
    query: str = "", country: str = "", page: int = 1
) -> Union[pd.DataFrame, None]:
    """
    This function takes in the query(species), country with cnt: at the start,
    page as page number. This will output a dataframe from the reqeust of the
    xeno canto api
    ---
    Args: query:str,country:str, page:str
    ---
    Returns: Dataframe consisting of the data retrieved from the api
    """
    try:
        response = requests.get(
            f"https://xeno-canto.org/api/2/recordings?query={query}{country}&page={page}",
            timeout=5,
        )
        if response.status_code != 200:
            print(f"Request failed with status code {response.status_code}")
            return None
        data = response.json()
        if "recordings" not in data:
            print("No recordings found")
            return None
        recordings = data["recordings"]
        df = pd.json_normalize(recordings)
        time.sleep(1)

        json_obj = json.loads(response.text)
        num_recordings = json_obj["numRecordings"]
        num_species = json_obj["numSpecies"]
        page = json_obj["page"]
        num_pages = json_obj["numPages"]

        print("Number of recordings: ", num_recordings),
        print("Number of species: ", num_species)
        print("Page: ", page)
        print("Number of pages: ", num_pages)
        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def get_queries(queries: list, country: str = "", page: int = 1) -> pd.DataFrame:
    concat_df = None
    for query in queries:
        query_df = get_query(query, country, page)
        if query_df is not None:
            if concat_df is None:
                concat_df = query_df
            else:
                concat_df = pd.concat([concat_df, query_df])
    return None


def download_files(df):
    """
    Downloads audio files from URLs specified in a Pandas DataFrame,
    saves them to a local 'data' folder, and appends the file path as
    a new column to the DataFrame.
    ---
    Args:
        df: Pandas DataFrame with 'file' and 'file-name' columns specifying
            the URLs and file names of the audio files to be downloaded.
    ---
    Returns:
        df: Pandas DataFrame with an additional 'file-path' column that
            contains the local file paths of the downloaded audio files.
    """
    for _, row in df.iterrows():
        url = row["file"]
        file_name = row["file-name"]
        response = requests.get(url)
        with open(row["file-path"], "wb") as f:
            f.write(response.content)
        print(f"File '{file_name}' downloaded to 'data' folder.")

    return df


def clean_file_name(file_name: str) -> str:
    """
    Replace any characters that are not alphanumeric, underscore,
    or period with an underscore
    ---
    Args: file name
    ---
    Returns : cleaned file name
    """
    return re.sub(r"[^\w\.]", "_", file_name)


def run_standalone() -> None:
    print("Getting query for Pycnonotus+goiavier")
    df_temp2 = get_query(query="Pycnonotus+goiavier")
    print("Getting query for Eudynamys scolopaceus")
    df_temp3 = get_query(query="Eudynamys scolopaceus")

    page = [1, 2, 3]
    df_int = pd.DataFrame()
    print("Getting query for Tringa+totanus")
    for i in page:
        df_temp = get_query(query="Tringa+totanus", page=i)
        df_int = pd.concat([df_int, df_temp])
    print("Concatenating the queries")
    df_int = pd.concat([df_int, df_temp2, df_temp3])
    print("Dropping missing files")
    df_int.drop(df_int.loc[df_int["file"] == ""].index, inplace=True)
    print("Cleaning file name.")
    df_int["file-name"] = df_int["file-name"].apply(clean_file_name)
    df_int["file-path"] = df_int["file-name"].apply(
        lambda x: "/".join(["data", "audio", x])
    )

    data_path = Path("./data/audio/")
    data_path.mkdir(parents=True, exist_ok=True)
    print("Downloading Audio Files")
    download_files(df_int)
    print("Audio files download complete")
    file_path = Path("./data/csv/")
    file_path.mkdir(parents=True, exist_ok=True)
    csv_path = file_path.joinpath("my_data.csv")
    print("Saving meta data")
    df_int.to_csv(csv_path, index=False)
    print("Meta data saved.")
    print("Data collection process completed")

    return None

    # Sample code
    # -----------
    # python -m src.data_module.data_collector


if __name__ == "__main__":
    run_standalone()
