
import zipfile
import requests

link = "http://images.cocodataset.org/zips/train2014.zip"

Print("Downloading data ...")
data = requests.get(link)
Print("Data Downloaded")
Print("Extracting data ...")
with zipfile.ZipFile(data, 'r') as zip_ref:
    zip_ref.extractall("Download/datacoco")