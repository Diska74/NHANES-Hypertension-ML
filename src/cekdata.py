import pandas as pd

demo = pd.read_sas("data/raw/DEMO_L.xpt")
bmx = pd.read_sas("data/raw/BMX_L.xpt")
bpxo = pd.read_sas("data/raw/BPXO_L.xpt")
bpq = pd.read_sas("data/raw/BPQ_L.xpt")

print("DEMO :", demo.shape)
print("BMX  :", bmx.shape)
print("BPXO :", bpxo.shape)
print("BPQ  :", bpq.shape)

print(demo[
    ["SEQN", "RIDAGEYR", "RIAGENDR", "DMDEDUC2", "INDFMPIR"]
].head())

print(bmx[
    ["SEQN", "BMXHT", "BMXWT", "BMXBMI", "BMXWAIST"]
].head())

print(bpxo[
    [
        "SEQN",
        "BPXOSY1",
        "BPXOSY2",
        "BPXOSY3",
        "BPXODI1",
        "BPXODI2",
        "BPXODI3"
    ]
].head())

print(bpq[
    ["SEQN", "BPQ020", "BPQ030", "BPQ150"]
].head())

demo = demo[demo["RIDAGEYR"] >= 18]

#AMBIL DATA YANG DIBUTUHKAN
#DEMO
demo = demo[
    [
        "SEQN",
        "RIDAGEYR",
        "RIAGENDR",
        "DMDEDUC2",
        "INDFMPIR"
    ]
]
#BMX
bmx = bmx[
    [
        "SEQN",
        "BMXHT",
        "BMXWT",
        "BMXBMI",
        "BMXWAIST"
    ]
]
#BPXO
bpxo = bpxo[
    [
        "SEQN",
        "BPXOSY1",
        "BPXOSY2",
        "BPXOSY3",
        "BPXODI1",
        "BPXODI2",
        "BPXODI3"
    ]
]
#BPQ
bpq = bpq[
    [
        "SEQN",
        "BPQ020",
        "BPQ030",
        "BPQ150"
    ]
]
#GABUNGAN DATA
df = demo.merge(bmx, on="SEQN", how="inner")
df = df.merge(bpxo, on="SEQN", how="inner")
df = df.merge(bpq, on="SEQN", how="inner")
print(df.shape)
print(df.head())

#rata rata hipertensi
df["SBP_mean"] = df[
    ["BPXOSY1", "BPXOSY2", "BPXOSY3"]
].mean(axis=1)

df["DBP_mean"] = df[
    ["BPXODI1", "BPXODI2", "BPXODI3"]
].mean(axis=1)
#Kelas Hipertensi
df = df[df["BPQ020"].isin([1, 2])]
df["Hypertension"] = df["BPQ020"].map({
    1: 1,
    2: 0
})
#Rename data 
df = df.rename(columns={
    "RIDAGEYR": "Age",
    "RIAGENDR": "Gender",
    "DMDEDUC2": "Education",
    "INDFMPIR": "Income_Poverty_Ratio",
    "BMXHT": "Height",
    "BMXWT": "Weight",
    "BMXBMI": "BMI",
    "BMXWAIST": "Waist"
})
#Buang Missing Value
features = [
    "Age",
    "Gender",
    "Education",
    "Income_Poverty_Ratio",
    "Height",
    "Weight",
    "BMI",
    "Waist"
]

df = df.dropna(
    subset=features + ["Hypertension"]
)
#Data Final
final_columns = [
    "SEQN",
    "Age",
    "Gender",
    "Education",
    "Income_Poverty_Ratio",
    "Height",
    "Weight",
    "BMI",
    "Waist",
    "Hypertension"
]

dataset_final = df[final_columns]
final_columns = [
    "SEQN",
    "Age",
    "Gender",
    "Education",
    "Income_Poverty_Ratio",
    "Height",
    "Weight",
    "BMI",
    "Waist",
    "Hypertension"
]

dataset_final = df[final_columns]
print(dataset_final.shape)
print(dataset_final.head())
print(dataset_final["Hypertension"].value_counts())
#download data csv
dataset_final.to_csv(
    "data/processed/dataset_hypertension.csv",
    index=False
)

print("Dataset berhasil disimpan!")