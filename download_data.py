
import zipfile
import requests

def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

print("Downloading data ...")

url = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip" 

download_url(url , "data/annot.zip")


print("Data Downloaded")
print("Extracting data ...")
with zipfile.ZipFile("data/annot.zip", 'r') as zip_ref:
    zip_ref.extractall("data/")