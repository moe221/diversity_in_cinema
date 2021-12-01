import pandas as pd
from tqdm import tqdm

from google.cloud import storage


from diversity_in_cinema.params import *
from diversity_in_cinema.utils import upload_file_to_gcp
from diversity_in_cinema.utils import gcp_file_names
from diversity_in_cinema.utils import final_stats



def create_stats_csv():

    file_names = gcp_file_names(BUCKET_NAME, "output")
    file_names.remove("summary.csv")

    for file in tqdm(file_names):

        # check if movie was already processed:
        client = storage.Client()
        processed_movies = [str(x).split("CSVs/")[1].\
            split("/statistics")[0].\
                replace("_", " ")\
                    for x in \
                        client.list_blobs(BUCKET_NAME_STREAMLIT, prefix="CSVs")]

        if file.replace(".csv", "").strip() in processed_movies:
            continue


        df = pd.read_csv(
            f"gs://{BUCKET_NAME}/output/{file}", index_col=None,)

        #replace spaces with underscores in movie name
        folder_name = file.strip().replace(".csv","").replace(" ","_")

        stats_df = final_stats(df)

        upload_file_to_gcp(stats_df, BUCKET_NAME_STREAMLIT,
                           f"CSVs/{folder_name}/statistics")


if __name__ == "__main__":

    create_stats_csv()
