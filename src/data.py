#data NHANES
import os
import urllib.request

# Folder penyimpanan
os.makedirs("data/raw", exist_ok=True)

files = {
    "DEMO_L.xpt":
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DEMO_L.xpt",

    "BMX_L.xpt":
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/BMX_L.xpt",

    "BPXO_L.xpt":
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/BPXO_L.xpt",

    "BPQ_L.xpt":
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/BPQ_L.xpt"
}

for filename, url in files.items():

    path = os.path.join("data/raw", filename)

    print(f"Downloading {filename}...")

    urllib.request.urlretrieve(url, path)

    print(f"Berhasil: {path}")

print("\nSemua file NHANES berhasil didownload.")
