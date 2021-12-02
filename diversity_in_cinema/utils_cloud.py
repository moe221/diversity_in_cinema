from PIL import Image
import numpy as np

import os
from io import BytesIO
import io

from google.cloud import storage
from google.api_core import page_iterator



def gcp_file_names(bucket_name, subfolders, file_type = ".csv"):

    """
    Function ro grab file names from a GCP bucket directory

    Parameters:

    bucket_name: Name of GCP bucket
    subfolders: complete subfolder path as a string where file names should
                be retrieved from in the format folder_1/folder_2/.../folder_n

    """

    client = storage.Client.create_anonymous_client()
    file_names = [str(x).split(f"{subfolders}/")[1].\
        split(".jpg")[0].\
            strip() + ".jpg" for x in \
                client.list_blobs(bucket_name, prefix=subfolders)]

    #blobs = client.list_blobs(bucket_name, prefix=subfolders)

    #bucket = client.bucket(bucket_name)

    #for blob in blobs: print(str(blob.name))

    return file_names

def _item_to_value(iterator, item):
    '''
    Helper fn for gcp_subdir_names below.
    '''
    return item


def gcp_subdir_names(bucket_name, prefix):
    '''
    Lists all folder names in a given bucket. Ripped straight from
    https://stackoverflow.com/questions/37074977/how-to-get-list-of-folders-in-a-given-bucket-using-google-cloud-api
    '''

    if prefix and not prefix.endswith('/'):
        prefix += '/'

    extra_params = {
        "projection": "noAcl",
        "prefix": prefix,
        "delimiter": '/'
    }

    gcs = storage.Client.create_anonymous_client()

    path = "/b/" + bucket_name + "/o"

    iterator = page_iterator.HTTPIterator(
        client=gcs,
        api_request=gcs._connection.api_request,
        path=path,
        items_key='prefixes',
        item_to_value=_item_to_value,
        extra_params=extra_params,
    )

    return [x[len(prefix):-1].lstrip() for x in iterator]

# def gcp_file_names(bucket_name, subfolders):

#     """
#     Function ro grab file names from a GCP bucket directory

#     Parameters:

#     bucket_name: Name of GCP bucket
#     subfolders: complete subfolder path as a string where file names should
#                 be retrieved from in the format folder_1/folder_2/.../folder_n

#     """

#     client = storage.Client.create_anonymous_client()
#     bucket = client.bucket(bucket_name)
#     # file_names = [str(x).split(f"{subfolders}/")[1].\
#     #     split(".csv")[0].\
#     #         strip() + ".csv" for x in \
#     #             bucket.list_blobs()]

#     return list(bucket.list_blobs(prefix=subfolders))


def upload_file_to_gcp(file, bucket_name, file_name, file_type="text/csv"):

    """
    Function to upload a file to a GCP bucket

    """

    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    bucket.blob(f'{file_name}').upload_from_string(file.to_csv(index=False),
                                                   file_type)
    print(
        f"Uploaded {file_name} data to: {bucket_name}/{file_name}")


def upload_image_to_gcp(image, bucket_name, image_name):

    """
    Function for uploading an image to a GCP Bucket

    """

    # Fix from https://stackoverflow.com/questions/60138697/typeerror-cannot-handle-this-data-type-1-1-3-f4
    im = image * 255
    im = im.astype(np.uint8)
    image = Image.fromarray(im)

    client = storage.Client.create_anonymous_client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(f"{image_name}")

    b = io.BytesIO()
    image.save(b, "jpeg")
    image.close()

    blob.upload_from_string(b.getvalue(), content_type="image/jpeg")

def bulk_get_image_pixels(files, bucket_name):

    client = storage.Client.create_anonymous_client()
    bucket = client.get_bucket(bucket_name)
    
    out = []

    for file in files:
        blob = bucket.get_blob(f"{file}")
        data = blob.download_as_string()
        img = Image.open(BytesIO(data))
        img = np.array(img)
        out.append(img)
    
    return out

def getImagePixels(file, bucket_name):

    client = storage.Client.create_anonymous_client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.get_blob(f"{file}")
    data = blob.download_as_string()
    img = Image.open(BytesIO(data))
    img = np.array(img)

    return img
