import pandas as pd
from tqdm import tqdm

from diversity_in_cinema.params import *
from diversity_in_cinema.utils import upload_file_to_gcp
from diversity_in_cinema.utils import gcp_file_names


def stats_function(df):

    # plug in Gabi's function here

    pass


def main():

    file_names = gcp_file_names(BUCKET_NAME, "output")
    file_names.remove("summary.csv")

    for file in tqdm(file_names):
        df = pd.read_csv(
            f"gs://{BUCKET_NAME}/output/{file}", index_col=None,)

        folder_name = file.strip().replace(".csv","").replace(" ","_")

        stats_df = stats_function(df)

        upload_file_to_gcp(stats_df, BUCKET_NAME_STREAMLIT,
                           f"CSVs/{folder_name}/statistics")
