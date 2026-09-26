import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    classification_report
)

df = pd.read_csv("data/processed/dataset_hypertension.csv")

print("Ukuran data:", df.shape)
print(df.head())
print(df.info())

#Data Cleaning
jumlah_duplikat = df.duplicated().sum()
print("\nJumlah duplikasi seluruh baris:")
print(jumlah_duplikat)
if jumlah_duplikat > 0:
    print("Menghapus baris duplikat...")
    df = df.drop_duplicates()
else:
    print("Tidak terdapat duplikasi.")
if "SEQN" in df.columns:
    jumlah_seqn_duplikat = df["SEQN"].duplicated().sum()
    print("\nJumlah SEQN duplikat:")
    print(jumlah_seqn_duplikat)
else:
    print("\nPERINGATAN: kolom SEQN tidak ditemukan.")
missing = df.isnull().sum()
print(missing)
missing_percentage = (
    df.isnull().sum() /
    len(df) *
    100
)
missing_table = pd.DataFrame({
    "Missing": df.isnull().sum(),
    "Persentase (%)": missing_percentage.round(2)

})
print("\nTabel missing value:")
print(missing_table)
if "Gender" in df.columns:
    print(
        df["Gender"]
        .value_counts(dropna=False)
        .sort_index()
    )
df["Education"] = df["Education"].replace(
    [9],
    np.nan
)
print("\nDistribusi Education:")
print(
    df["Education"]
    .value_counts(dropna=False)
    .sort_index()
)
df = df.dropna().copy()
print(
    df["Hypertension"]
    .value_counts(dropna=False)
    .sort_index()
)

#Pemeriksaan Data Tidak Valid
if "Age" in df.columns:
    print("\nAge:")
    print(
        "Nilai minimum:",
        df["Age"].min()
    )
    print(
        "Nilai maksimum:",
        df["Age"].max()
    )
    invalid_age = (
        (df["Age"] < 0) |
        (df["Age"] > 120)
    ).sum()
    print(
        "Jumlah Age tidak valid:",
        invalid_age
    )
if "Height" in df.columns:
    print("\nHeight:")
    print(
        "Nilai minimum:",
        df["Height"].min()
    )
    print(
        "Nilai maksimum:",
        df["Height"].max()
    )
    invalid_height = (
        (df["Height"] <= 0)
    ).sum()
    print(
        "Jumlah Height <= 0:",
        invalid_height
    )
if "Weight" in df.columns:
    print("\nWeight:")
    print(
        "Nilai minimum:",
        df["Weight"].min()
    )
    print(
        "Nilai maksimum:",
        df["Weight"].max()
    )
    invalid_weight = (
        (df["Weight"] <= 0)
    ).sum()
    print(
        "Jumlah Weight <= 0:",
        invalid_weight
    )
if "BMI" in df.columns:
    print("\nBMI:")
    print(
        "Nilai minimum:",
        df["BMI"].min()
    )
    print(
        "Nilai maksimum:",
        df["BMI"].max()
    )
    invalid_bmi = (
        (df["BMI"] <= 0)
    ).sum()
    print(
        "Jumlah BMI <= 0:",
        invalid_bmi
    )
if "Waist" in df.columns:
    print("\nWaist:")
    print(
        "Nilai minimum:",
        df["Waist"].min()
    )
    print(
        "Nilai maksimum:",
        df["Waist"].max()
    )
    invalid_waist = (
        (df["Waist"] <= 0)
    ).sum()
    print(
        "Jumlah Waist <= 0:",
        invalid_waist
    )
#Cek Outlier
numerical_features = [
    "Age",
    "Income_Poverty_Ratio",
    "Height",
    "Weight",
    "BMI",
    "Waist"
]
outlier_results = []
for col in numerical_features:
    if col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_mask = (
            (df[col] < lower_bound) |
            (df[col] > upper_bound)
        )
        jumlah_outlier = outlier_mask.sum()
        persentase_outlier = (
            jumlah_outlier /
            len(df) *
            100
        )
        outlier_results.append({
            "Variabel": col,
            "Q1": Q1,
            "Q3": Q3,
            "Lower Bound": lower_bound,
            "Upper Bound": upper_bound,
            "Jumlah Outlier": jumlah_outlier,
            "Persentase (%)": persentase_outlier
        })
outlier_table = pd.DataFrame(
    outlier_results
)
print(
    outlier_table.round(2)
)
for i, col in enumerate(numerical_features):
    plt.subplot(2,3,i+1)
    sns.boxplot(y=df[col])
    plt.title(col)
plt.tight_layout()

if all(
    col in df.columns
    for col in ["Height", "Weight", "BMI"]
):
 # Tinggi dalam cm → meter
    calculated_bmi = (
        df["Weight"] /
        (df["Height"] / 100) ** 2
    )
    bmi_difference = (
        abs(calculated_bmi - df["BMI"])
    )
    print(
        "Median selisih BMI:",
        bmi_difference.median()
    )
    print(
        "Maksimum selisih BMI:",
        bmi_difference.max()
    )
    print(
        "\nCatatan:"
    )
    print(
        "Pemeriksaan ini digunakan untuk melihat "
        "konsistensi Height, Weight, dan BMI."
    )
print(
    df["Hypertension"]
    .value_counts()
    .sort_index()
)
print("\nProporsi kelas:")
print(
    df["Hypertension"]
    .value_counts(
        normalize=True
    )
    .sort_index()
)
plt.figure(figsize=(7, 5))
df["Hypertension"].value_counts().sort_index().plot(
    kind="bar"
)
plt.title(
    "Distribusi Status Hipertensi"
)
plt.xlabel(
    "Status Hipertensi"
)
plt.ylabel(
    "Jumlah Responden"
)
plt.xticks(
    ticks=[0, 1],
    labels=[
        "Tidak Hipertensi",
        "Hipertensi"
    ],
    rotation=0
)
plt.tight_layout()
print(
    "Ukuran dataset:",
    df.shape
)
print(
    "\nMissing value:"
)
print(
    df.isnull().sum()
)
# Menetukan X dan Y
X = df.drop(
    columns=[
        "SEQN",
        "Hypertension"
    ]
)
y = df["Hypertension"]
print(
    "\nFitur:"
)
print(
    X.columns.tolist()
)
print(
    "\nTarget:"
)
print(
    y.name
)
#Train-Test Split 
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
print(
    "X_train:",
    X_train.shape
)
print(
    "X_test:",
    X_test.shape
)
print(
    "y_train:",
    y_train.shape
)
print(
    "y_test:",
    y_test.shape
)
#Scalling 
numerical_features = [
    "Age",
    "Income_Poverty_Ratio",
    "Height",
    "Weight",
    "BMI",
    "Waist"
]
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numerical_features] = (
    scaler.fit_transform(
        X_train[numerical_features]
    )
)
X_test_scaled[numerical_features] = (
    scaler.transform(
        X_test[numerical_features]
    )
)
print(
    "Scaling selesai."
)
#Logistic Regression
logreg = LogisticRegression(
    max_iter=1000,
    random_state=42
)
logreg.fit(
    X_train_scaled,
    y_train
)
pred_logreg = logreg.predict(
    X_test_scaled
)
prob_logreg = logreg.predict_proba(
    X_test_scaled
)[:, 1]
acc_logreg = accuracy_score(
    y_test,
    pred_logreg
)
precision_logreg = precision_score(
    y_test,
    pred_logreg
)
recall_logreg = recall_score(
    y_test,
    pred_logreg
)
f1_logreg = f1_score(
    y_test,
    pred_logreg
)
auc_logreg = roc_auc_score(
    y_test,
    prob_logreg
)
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        pred_logreg
    )
)
print(
    "Accuracy :",
    round(acc_logreg, 4)
)
#Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf.fit(
    X_train,
    y_train
)
pred_rf = rf.predict(
    X_test
)
prob_rf = rf.predict_proba(
    X_test
)[:, 1]
acc_rf = accuracy_score(
    y_test,
    pred_rf
)
precision_rf = precision_score(
    y_test,
    pred_rf
)
recall_rf = recall_score(
    y_test,
    pred_rf
)
f1_rf = f1_score(
    y_test,
    pred_rf
)
auc_rf = roc_auc_score(
    y_test,
    prob_rf
)
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        pred_rf
    )
)
print(
    "Accuracy :",
    round(acc_rf, 4)
)
#XGBoost
xgb = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss"
)
xgb.fit(
    X_train,
    y_train
)
pred_xgb = xgb.predict(
    X_test
)
prob_xgb = xgb.predict_proba(
    X_test
)[:, 1]
acc_xgb = accuracy_score(
    y_test,
    pred_xgb
)
precision_xgb = precision_score(
    y_test,
    pred_xgb
)
recall_xgb = recall_score(
    y_test,
    pred_xgb
)
f1_xgb = f1_score(
    y_test,
    pred_xgb
)
auc_xgb = roc_auc_score(
    y_test,
    prob_xgb
)
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        pred_xgb
    )
)
print(
    "Accuracy :",
    round(acc_xgb, 4)
)
hasil_awal = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "XGBoost"
    ],
    "Accuracy": [
        acc_logreg,
        acc_rf,
        acc_xgb
    ],
    "Precision": [
        precision_logreg,
        precision_rf,
        precision_xgb
    ],
    "Recall": [
        recall_logreg,
        recall_rf,
        recall_xgb
    ],
    "F1-Score": [
        f1_logreg,
        f1_rf,
        f1_xgb
    ],
    "ROC-AUC": [
        auc_logreg,
        auc_rf,
        auc_xgb
    ]
})
print(
    "PERBANDINGAN MODEL SEBELUM TUNING"
)
print(
    hasil_awal.round(4)
)

os.makedirs("results", exist_ok=True)

#FIGURE EXPORT
import os

FIGURE_DIR = "figures"
os.makedirs(FIGURE_DIR, exist_ok=True)

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

def save_figure(filename):
    plt.savefig(
        os.path.join(FIGURE_DIR, filename + ".pdf"),
        format="pdf",
        bbox_inches="tight"
    )

#ROC CURVE
fpr_logreg, tpr_logreg, _ = roc_curve(
    y_test,
    prob_logreg
)
fpr_rf, tpr_rf, _ = roc_curve(
    y_test,
    prob_rf
)
fpr_xgb, tpr_xgb, _ = roc_curve(
    y_test,
    prob_xgb
)
plt.figure(
    figsize=(8, 6)
)
plt.plot(
    fpr_logreg,
    tpr_logreg,
    label=f"Logistic Regression (AUC = {auc_logreg:.3f})"
)
plt.plot(
    fpr_rf,
    tpr_rf,
    label=f"Random Forest (AUC = {auc_rf:.3f})"
)
plt.plot(
    fpr_xgb,
    tpr_xgb,
    label=f"XGBoost (AUC = {auc_xgb:.3f})"
)
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)
plt.xlabel(
    "False Positive Rate"
)
plt.ylabel(
    "True Positive Rate"
)
plt.title(
    "ROC Curve before Tuning"
)
plt.legend()
plt.tight_layout()

save_figure("Figure_ROC_baseline")

#plt.show()

#Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV
param_rf = {
    "n_estimators": [
        100,
        200,
        300
    ],
    "max_depth": [
        None,
        10,
        20
    ],
    "min_samples_split": [
        2,
        5
    ],
    "min_samples_leaf": [
        1,
        2
    ]
}
rf_base = RandomForestClassifier(
    random_state=42
)
grid_rf = GridSearchCV(
    estimator=rf_base,
    param_grid=param_rf,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)
grid_rf.fit(
    X_train,
    y_train
)
print(
    "\nBest Parameters Random Forest:"
)
print(
    grid_rf.best_params_
)
print(
    "\nBest CV ROC AUC:"
)
print(
    round(
        grid_rf.best_score_,
        4
    )
)
best_rf = grid_rf.best_estimator_
pred_rf_tuned = best_rf.predict(
    X_test
)
prob_rf_tuned = best_rf.predict_proba(
    X_test
)[:, 1]
acc_rf_tuned = accuracy_score(
    y_test,
    pred_rf_tuned
)
precision_rf_tuned = precision_score(
    y_test,
    pred_rf_tuned
)
recall_rf_tuned = recall_score(
    y_test,
    pred_rf_tuned
)
f1_rf_tuned = f1_score(
    y_test,
    pred_rf_tuned
)
auc_rf_tuned = roc_auc_score(
    y_test,
    prob_rf_tuned
)
print(
    classification_report(
        y_test,
        pred_rf_tuned
    )
)
print(
    "Accuracy :",
    round(acc_rf_tuned, 4)
)

param_xgb = {
    "n_estimators": [
        100,
        200,
        300
    ],
    "max_depth": [
        3,
        5
    ],
    "learning_rate": [
        0.05,
        0.1
    ],
    "subsample": [
        0.8,
        1.0
    ]
}
xgb_base = XGBClassifier(
    random_state=42,
    eval_metric="logloss"
)
grid_xgb = GridSearchCV(
    estimator=xgb_base,
    param_grid=param_xgb,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)
grid_xgb.fit(
    X_train,
    y_train
)
print(
    "\nBest Parameters XGBoost:"
)
print(
    grid_xgb.best_params_
)
print(
    "\nBest CV ROC AUC:"
)
print(
    round(
        grid_xgb.best_score_,
        4
    )
)
best_xgb = grid_xgb.best_estimator_
pred_xgb_tuned = best_xgb.predict(
    X_test
)
prob_xgb_tuned = best_xgb.predict_proba(
    X_test
)[:, 1]
acc_xgb_tuned = accuracy_score(
    y_test,
    pred_xgb_tuned
)
precision_xgb_tuned = precision_score(
    y_test,
    pred_xgb_tuned
)
recall_xgb_tuned = recall_score(
    y_test,
    pred_xgb_tuned
)
f1_xgb_tuned = f1_score(
    y_test,
    pred_xgb_tuned
)
auc_xgb_tuned = roc_auc_score(
    y_test,
    prob_xgb_tuned
)
print(
    classification_report(
        y_test,
        pred_xgb_tuned
    )
)
print(
    "Accuracy :",
    round(acc_xgb_tuned, 4)
)

prob_xgb = best_xgb.predict_proba(X_test.iloc[[0]])[0, 1]
print(f"XGBoost predicted probability: {prob_xgb:.3f}")

#Evaluasi 2
hasil_akhir = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "Random Forest Tuned",
        "XGBoost",
        "XGBoost Tuned"
    ],
    "Accuracy": [
        acc_logreg,
        acc_rf,
        acc_rf_tuned,
        acc_xgb,
        acc_xgb_tuned
    ],
    "Precision": [
        precision_logreg,
        precision_rf,
        precision_rf_tuned,
        precision_xgb,
        precision_xgb_tuned
    ],
    "Recall": [
        recall_logreg,
        recall_rf,
        recall_rf_tuned,
        recall_xgb,
        recall_xgb_tuned
    ],
    "F1-Score": [
        f1_logreg,
        f1_rf,
        f1_rf_tuned,
        f1_xgb,
        f1_xgb_tuned
    ],
    "ROC-AUC": [
        auc_logreg,
        auc_rf,
        auc_rf_tuned,
        auc_xgb,
        auc_xgb_tuned
    ]
})
print("\n" + "=" * 70)
print(
    "HASIL AKHIR SEMUA MODEL"
)
print("=" * 70)
print(
    hasil_akhir.round(4)
)

#SHAP Random Forest
import shap

explainer_rf = shap.Explainer(best_rf, X_train)
shap_values_rf = explainer_rf(X_test)
print("SHAP values shape :", shap_values_rf.values.shape)
print("Base values shape :", shap_values_rf.base_values.shape)
print("Data shape        :", shap_values_rf.data.shape)
shap_values_rf_class1 = shap_values_rf[:, :, 1]
# Beeswarm
plt.figure(figsize=(10, 7))

shap.plots.beeswarm( 
    shap_values_rf_class1, 
    max_display=len(X_test.columns), 
    show=False 
)

plt.title("SHAP Beeswarm Plot – Tuned Random Forest") 
plt.tight_layout()

save_figure("Figure_SHAP_RF_beeswarm")

plt.show()
# Bar
plt.figure(figsize=(10, 7))

shap.plots.bar( 
    shap_values_rf_class1, 
    max_display=len(X_test.columns), 
    show=False 
)

plt.title("SHAP Feature Importance - Tuned Random Forest") 
plt.tight_layout()

save_figure("Figure_SHAP_RF_bar")

plt.show()
# Waterfall
plt.figure(figsize=(10, 7))

shap.plots.waterfall( 
    shap_values_rf_class1[0], 
    max_display=len(X_test.columns), 
    show=False 
)

plt.title("SHAP Waterfall - Tuned Random Forest") 
plt.tight_layout()

save_figure("Figure_SHAP_RF_waterfall")

plt.show()
#SHAP XGBoost
explainer_xgb = shap.TreeExplainer(
    best_xgb,
    feature_perturbation="tree_path_dependent"
)
shap_values_xgb = explainer_xgb(X_test)

print("SHAP values shape :", shap_values_xgb.values.shape)
print("Base values shape :", shap_values_xgb.base_values.shape)
print("Data shape        :", shap_values_xgb.data.shape)
# Beeswarm
plt.figure(figsize=(10, 7))

shap.plots.beeswarm( 
    shap_values_xgb, 
    max_display=len(X_test.columns), 
    show=False 
)

plt.title("SHAP Beeswarm - Tuned XGBoost") 
plt.tight_layout()

save_figure("Figure_SHAP_XGB_beeswarm")

plt.show()
# Bar
plt.figure(figsize=(10, 7))

shap.plots.bar( 
    shap_values_xgb, 
    max_display=len(X_test.columns), 
    show=False 
)

plt.title("SHAP Feature Importance - Tuned XGBoost") 
plt.tight_layout()

save_figure("Figure_SHAP_XGB_bar")

plt.show()
# Waterfall
plt.figure(figsize=(10, 7))

shap.plots.waterfall( 
    shap_values_xgb[0], 
    max_display=len(X_test.columns), 
    show=False 
)

plt.title("SHAP Waterfall - Tuned XGBoost") 
plt.tight_layout()

save_figure("Figure_SHAP_XGB_waterfall")

plt.show()

from scipy.stats import spearmanr
from scipy.special import ndtr
from sklearn.utils import resample



# A. QUANTITATIVE CROSS-MODEL SHAP AGREEMENT

print("\n" + "=" * 70)
print("A. QUANTITATIVE CROSS-MODEL SHAP AGREEMENT")
print("=" * 70)

# ------------------------------------------------------------
# 1. Pastikan SHAP XGBoost menggunakan class 1 jika output 3D
# ------------------------------------------------------------

if shap_values_xgb.values.ndim == 3:
    shap_values_xgb_class1 = shap_values_xgb[:, :, 1]
else:
    shap_values_xgb_class1 = shap_values_xgb

# ------------------------------------------------------------
# 2. Hitung mean absolute SHAP
# ------------------------------------------------------------

mean_abs_shap_rf = np.abs(
    shap_values_rf_class1.values
).mean(axis=0)

mean_abs_shap_xgb = np.abs(
    shap_values_xgb_class1.values
).mean(axis=0)

features_shap = X_test.columns.tolist()

shap_comparison = pd.DataFrame({
    "Feature": features_shap,
    "RF_Mean_Abs_SHAP": mean_abs_shap_rf,
    "XGB_Mean_Abs_SHAP": mean_abs_shap_xgb
})

# ------------------------------------------------------------
# 3. Ranking SHAP masing-masing model
# ------------------------------------------------------------

shap_comparison["RF_Rank"] = (
    shap_comparison["RF_Mean_Abs_SHAP"]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)

shap_comparison["XGB_Rank"] = (
    shap_comparison["XGB_Mean_Abs_SHAP"]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)

# Urut berdasarkan ranking RF
shap_comparison = shap_comparison.sort_values(
    "RF_Rank"
).reset_index(drop=True)

print("\nRanking SHAP:")
print(
    shap_comparison.round(4)
)

# ------------------------------------------------------------
# 4. Spearman correlation antara ranking RF dan XGBoost
# ------------------------------------------------------------

spearman_shap, p_shap = spearmanr(
    shap_comparison["RF_Rank"],
    shap_comparison["XGB_Rank"]
)

print("\nCross-model SHAP agreement:")
print(
    f"Spearman rho = {spearman_shap:.4f}"
)

print(
    f"p-value = {p_shap:.4f}"
)

# ------------------------------------------------------------
# 5. Simpan tabel SHAP
# ------------------------------------------------------------

shap_comparison.to_csv(
    "results/shap_cross_model_comparison.csv",
    index=False
)

print(
    "\nTabel SHAP disimpan sebagai:"
    " shap_cross_model_comparison.csv"
)


# ============================================================
# B. BOOTSTRAP 95% CONFIDENCE INTERVAL
# ============================================================

print("\n" + "=" * 70)
print("B. BOOTSTRAP 95% CONFIDENCE INTERVAL")
print("=" * 70)


def bootstrap_metrics(
    y_true,
    probabilities,
    predictions,
    n_bootstrap=2000,
    random_state=42
):

    rng = np.random.RandomState(random_state)

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    predictions = np.asarray(predictions)

    n = len(y_true)

    auc_values = []
    recall_values = []
    f1_values = []

    for _ in range(n_bootstrap):

        indices = rng.randint(
            0,
            n,
            n
        )

        y_boot = y_true[indices]
        prob_boot = probabilities[indices]
        pred_boot = predictions[indices]

        # Skip bootstrap samples containing only one class
        if len(np.unique(y_boot)) < 2:
            continue

        auc_values.append(
            roc_auc_score(
                y_boot,
                prob_boot
            )
        )

        recall_values.append(
            recall_score(
                y_boot,
                pred_boot,
                zero_division=0
            )
        )

        f1_values.append(
            f1_score(
                y_boot,
                pred_boot,
                zero_division=0
            )
        )

    def ci(values):

        lower = np.percentile(
            values,
            2.5
        )

        upper = np.percentile(
            values,
            97.5
        )

        return lower, upper

    return {
        "ROC-AUC": (
            np.mean(auc_values),
            *ci(auc_values)
        ),

        "Recall": (
            np.mean(recall_values),
            *ci(recall_values)
        ),

        "F1-Score": (
            np.mean(f1_values),
            *ci(f1_values)
        )
    }


# ------------------------------------------------------------
# Predictions dari model
# ------------------------------------------------------------

predictions_models = {

    "Logistic Regression": (
        prob_logreg,
        pred_logreg
    ),

    "Random Forest Tuned": (
        prob_rf_tuned,
        pred_rf_tuned
    ),

    "XGBoost Tuned": (
        prob_xgb_tuned,
        pred_xgb_tuned
    )
}


bootstrap_results = []


for model_name, (
    probabilities,
    predictions
) in predictions_models.items():

    metrics = bootstrap_metrics(
        y_test,
        probabilities,
        predictions,
        n_bootstrap=2000,
        random_state=42
    )

    for metric_name, (
        estimate,
        lower,
        upper
    ) in metrics.items():

        bootstrap_results.append({

            "Model": model_name,

            "Metric": metric_name,

            "Estimate": estimate,

            "CI_Lower": lower,

            "CI_Upper": upper

        })


bootstrap_table = pd.DataFrame(
    bootstrap_results
)


print("\nBootstrap 95% CI:")
print(
    bootstrap_table.round(4)
)


bootstrap_table.to_csv(
    "results/bootstrap_95CI_results.csv",
    index=False
)

print(
    "\nHasil bootstrap disimpan sebagai:"
    " bootstrap_95CI_results.csv"
)


# ============================================================
# C. STATISTICAL COMPARISON OF ROC-AUC
#    DeLong Test + Holm Correction
# ============================================================

from scipy.stats import norm


# ------------------------------------------------------------
# Fungsi menghitung midrank
# ------------------------------------------------------------
def compute_midrank(x):
    x = np.asarray(x)
    order = np.argsort(x)
    sorted_x = x[order]

    midranks = np.empty(len(x), dtype=float)

    i = 0
    while i < len(x):
        j = i

        while j < len(x) and sorted_x[j] == sorted_x[i]:
            j += 1

        midrank = 0.5 * (i + j - 1) + 1
        midranks[order[i:j]] = midrank

        i = j

    return midranks


# ------------------------------------------------------------
# Fast DeLong
# ------------------------------------------------------------
def fast_delong(predictions, labels):
    """
    DeLong test for correlated ROC-AUCs.

    predictions:
        array shape = (n_models, n_samples)

    labels:
        binary labels (0/1)
    """

    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    # Positive cases harus berada di depan
    order = np.argsort(-labels)

    labels_sorted = labels[order]
    predictions_sorted = predictions[:, order]

    m = int(np.sum(labels_sorted == 1))
    n = int(np.sum(labels_sorted == 0))

    if m == 0 or n == 0:
        raise ValueError(
            "DeLong test membutuhkan minimal satu observation "
            "pada masing-masing kelas."
        )

    k = predictions_sorted.shape[0]

    tx = np.zeros((k, m))
    ty = np.zeros((k, n))
    tz = np.zeros((k, m + n))

    for r in range(k):
        tx[r, :] = compute_midrank(
            predictions_sorted[r, :m]
        )

        ty[r, :] = compute_midrank(
            predictions_sorted[r, m:]
        )

        tz[r, :] = compute_midrank(
            predictions_sorted[r, :]
        )

    # AUC setiap model
    aucs = (
        tz[:, :m].sum(axis=1) / (m * n)
        - (m + 1) / (2 * n)
    )

    # Structural components
    v01 = (
        tz[:, :m] - tx
    ) / n

    v10 = 1 - (
        tz[:, m:] - ty
    ) / m

    # Covariance matrix
    sx = np.cov(v01)
    sy = np.cov(v10)

    # Jika hanya terdapat satu model
    if k == 1:
        sx = np.array([[sx]])
        sy = np.array([[sy]])

    covariance = (
        sx / m
        + sy / n
    )

    return aucs, covariance


# ------------------------------------------------------------
# Fungsi p-value untuk perbandingan dua AUC
# ------------------------------------------------------------
def delong_pairwise_test(
    aucs,
    covariance,
    i,
    j
):

    auc_difference = aucs[i] - aucs[j]

    variance_difference = (
        covariance[i, i]
        + covariance[j, j]
        - 2 * covariance[i, j]
    )

    # Perlindungan terhadap numerical precision
    variance_difference = max(
        variance_difference,
        0
    )

    if variance_difference == 0:
        z = np.nan
        p_value = np.nan
    else:
        z = (
            auc_difference
            / np.sqrt(variance_difference)
        )

        p_value = 2 * norm.sf(abs(z))

    return (
        auc_difference,
        z,
        p_value
    )


# ------------------------------------------------------------
# Predicted probabilities dari tiga final models
# ------------------------------------------------------------
model_names = [
    "Logistic Regression",
    "Random Forest Tuned",
    "XGBoost Tuned"
]

predictions = np.vstack([
    prob_logreg,
    prob_rf_tuned,
    prob_xgb_tuned
])


# ------------------------------------------------------------
# Hitung DeLong
# ------------------------------------------------------------
aucs_delong, covariance_delong = fast_delong(
    predictions,
    y_test.values
)


# ------------------------------------------------------------
# Pairwise comparisons
# ------------------------------------------------------------
comparisons = [
    (0, 1),
    (0, 2),
    (1, 2)
]

delong_results = []

for i, j in comparisons:

    auc_difference, z, p_value = delong_pairwise_test(
        aucs_delong,
        covariance_delong,
        i,
        j
    )

    delong_results.append({
        "Model 1": model_names[i],
        "Model 2": model_names[j],
        "AUC 1": aucs_delong[i],
        "AUC 2": aucs_delong[j],
        "AUC Difference": auc_difference,
        "Z": z,
        "p-value": p_value
    })


delong_df = pd.DataFrame(
    delong_results
)


# ============================================================
# Holm correction for multiple comparisons
# ============================================================

def holm_correction(p_values):
    """
    Holm-Bonferroni correction.
    """

    p_values = np.asarray(
        p_values,
        dtype=float
    )

    m = len(p_values)

    order = np.argsort(p_values)

    adjusted = np.empty(m)

    previous = 0

    for rank, idx in enumerate(order):

        adjusted_value = (
            m - rank
        ) * p_values[idx]

        # adjusted p-values harus monotonik
        adjusted_value = max(
            adjusted_value,
            previous
        )

        adjusted_value = min(
            adjusted_value,
            1.0
        )

        adjusted[idx] = adjusted_value

        previous = adjusted_value

    return adjusted


delong_df[
    "p-value Holm"
] = holm_correction(
    delong_df["p-value"].values
)


# ------------------------------------------------------------
# Statistical significance
# ------------------------------------------------------------
alpha = 0.05

delong_df["Significant"] = np.where(
    delong_df["p-value Holm"] < alpha,
    "Yes",
    "No"
)


# ------------------------------------------------------------
# Rapikan angka
# ------------------------------------------------------------
delong_display = delong_df.copy()

for col in [
    "AUC 1",
    "AUC 2",
    "AUC Difference",
    "Z",
    "p-value",
    "p-value Holm"
]:

    delong_display[col] = (
        delong_display[col]
        .round(4)
    )


# ------------------------------------------------------------
# Print hasil
# ------------------------------------------------------------
print("\n" + "=" * 70)
print(
    "C. STATISTICAL COMPARISON OF ROC-AUC"
)
print("=" * 70)

print("\nDeLong test with Holm correction:")
print(
    delong_display.to_string(
        index=True
    )
)


# ------------------------------------------------------------
# Simpan hasil
# ------------------------------------------------------------
delong_df.to_csv(
    "results/delong_auc_comparison.csv",
    index=False
)

print(
    "\nHasil DeLong disimpan sebagai: "
    "results/delong_auc_comparison.csv"
)
# ============================================================
# D. CORRELATION AMONG ANTHROPOMETRIC VARIABLES
# ============================================================

from scipy.stats import spearmanr
import seaborn as sns
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Variabel antropometrik
# ------------------------------------------------------------
anthropometric_vars = [
    "Height",
    "Weight",
    "BMI",
    "Waist"
]

# Spearman correlation matrix

corr_matrix = (
    X_test[anthropometric_vars]
    .corr(method="spearman")
)


# ------------------------------------------------------------
# Print correlation matrix
# ------------------------------------------------------------
print("\n" + "=" * 70)
print(
    "D. CORRELATION AMONG ANTHROPOMETRIC VARIABLES"
)
print("=" * 70)

print("\nSpearman correlation matrix:")
print(
    corr_matrix.round(4)
)


plt.figure(
    figsize=(8, 6)
)

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    vmin=-1,
    vmax=1,
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={
        "label": "Spearman Correlation"
    }
)

plt.title(
    "Spearman Correlation Among Anthropometric Predictors"
)

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# Simpan correlation matrix
# ------------------------------------------------------------
corr_matrix.to_csv(
    "results/spearman_anthropometric_correlation.csv"
)

print(
    "\nCorrelation matrix disimpan sebagai: "
    "results/spearman_anthropometric_correlation.csv"
)
# ============================================================
# RINGKASAN ANALISIS TAMBAHAN
# ============================================================

print("\n" + "=" * 70)
print(
    "RINGKASAN ANALISIS TAMBAHAN"
)
print("=" * 70)

print(
    f"\nA. SHAP agreement:"
    f" Spearman rho = {spearman_shap:.4f},"
    f" p = {p_shap:.4f}"
)

print(
    "\nB. Bootstrap:"
    "\n   95% confidence intervals telah dihitung"
)

print(
    "\nC. DeLong:"
    "\n   Perbandingan ROC-AUC antar-model telah dihitung"
)

print(
    "\nD. Anthropometric correlation:"
    "\n   Spearman correlation matrix telah dihitung"
)

print("\nSemua analisis tambahan selesai.")