
import zipfile
import requests
import os

def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

print("Downloading data ...")

url_images = "http://images.cocodataset.org/zips/train2014.zip" 
url_annot = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip" 

download_url(url_images , "data/images.zip")
download_url(url_annot , "data/annot.zip")


print("Data Downloaded")
print("Extracting data ...")
with zipfile.ZipFile("data/images.zip", 'r') as zip_ref:
    zip_ref.extractall("data/")
with zipfile.ZipFile("data/annot.zip", 'r') as zip_ref:
    zip_ref.extractall("data/")

os.remove("data/images.zip")
os.remove("data/annot.zip")