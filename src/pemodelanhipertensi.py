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

df = pd.read_csv("dataset_hypertension.csv")

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

#FIGURE EXPORT
import os

FIGURE_DIR = "figures_pdf"
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
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 9.5,
    "pdf.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ============================================================
# ROC CURVE BEFORE TUNING
# ============================================================

fpr_logreg, tpr_logreg, _ = roc_curve(y_test, prob_logreg)
fpr_rf, tpr_rf, _ = roc_curve(y_test, prob_rf)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, prob_xgb)

# Figure sedikit lebih lebar agar legend tidak bertumpuk
fig, ax = plt.subplots(figsize=(5.2, 4.3))

# Logistic Regression
ax.plot(
    fpr_logreg,
    tpr_logreg,
    lw=1.8,
    color="#1f77b4",
    label=f"Logistic Regression (AUC = {auc_logreg:.4f})"
)

# Random Forest
ax.plot(
    fpr_rf,
    tpr_rf,
    lw=1.8,
    color="#d62728",
    label=f"Random Forest (AUC = {auc_rf:.4f})"
)

# XGBoost
ax.plot(
    fpr_xgb,
    tpr_xgb,
    lw=1.8,
    color="#2ca02c",
    label=f"XGBoost (AUC = {auc_xgb:.4f})"
)

# Garis diagonal / random classifier
ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="black",
    lw=1.0,
    label="Random classifier"
)

# Label sumbu
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")

# Batas sumbu
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)

# Tick
ax.tick_params(axis="both", labelsize=10)

# Legend
ax.legend(
    loc="lower right",
    frameon=False,
    fontsize=9.5,
    handlelength=2.5,
    handletextpad=0.6,
    labelspacing=0.5
)

# Layout
fig.tight_layout()

# ============================================================
# SAVE
# ============================================================

# PDF vector
fig.savefig(
    "fig2_roc_before_tuning.pdf",
    format="pdf",
    bbox_inches="tight"
)

# PNG 300 dpi
fig.savefig(
    "fig2_roc_before_tuning.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

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

# Fungsi untuk menyimpan figure dalam format PDF
def save_figure(name):
    plt.savefig(
        f"{name}.pdf",
        format="pdf",
        bbox_inches="tight"
    )

#SHAP Random Forest
# ============================================================
# SHAP ANALYSIS
# Random Forest and XGBoost
# Publication-ready figures
# ============================================================

import matplotlib.pyplot as plt
import shap
import numpy as np

# ============================================================
# GLOBAL FIGURE SETTINGS
# ============================================================

FS = 11  # Main manuscript font size

plt.rcParams.update({
    "font.family": "serif",
    "font.size": FS,

    # Axis
    "axes.labelsize": FS,
    "axes.titlesize": FS,

    # Ticks
    "xtick.labelsize": FS,
    "ytick.labelsize": FS,

    # Legend
    "legend.fontsize": FS,

    # Vector PDF
    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    # SVG font type if needed
    "svg.fonttype": "none",

    # Prevent unwanted clipping
    "figure.autolayout": False
})


# ============================================================
# PUBLICATION-FRIENDLY FEATURE LABELS
# ============================================================

PRETTY = {
    "Age": "Age",
    "Gender": "Gender",
    "Education": "Education",
    "Income_Poverty_Ratio": "Income-to-poverty ratio",
    "Height": "Height",
    "Weight": "Weight",
    "BMI": "BMI",
    "Waist": "Waist circumference"
}

# Original feature order
FEATURES = list(X_test.columns)

# Publication-friendly names
NAMES = [PRETTY.get(col, col) for col in FEATURES]


# ============================================================
# GENERAL FIGURE STYLING FUNCTION
# ============================================================

def style_fig(fig, fs=FS):
    """
    Apply consistent manuscript-sized fonts to all
    internal figure elements.
    """

    for ax in fig.axes:

        # ----------------------------------------------------
        # Axis labels
        # ----------------------------------------------------
        ax.xaxis.label.set_fontsize(fs)
        ax.yaxis.label.set_fontsize(fs)

        # ----------------------------------------------------
        # Tick labels
        # ----------------------------------------------------
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=fs
        )

        ax.tick_params(
            axis="both",
            which="minor",
            labelsize=fs
        )

        # ----------------------------------------------------
        # Existing text / annotations
        # ----------------------------------------------------
        for txt in ax.texts:
            txt.set_fontsize(fs)

        # ----------------------------------------------------
        # X and Y tick labels
        # ----------------------------------------------------
        for txt in ax.get_xticklabels():
            txt.set_fontsize(fs)

        for txt in ax.get_yticklabels():
            txt.set_fontsize(fs)

        # ----------------------------------------------------
        # Axis title, if SHAP creates one
        # ----------------------------------------------------
        ax.title.set_fontsize(fs)

    # --------------------------------------------------------
    # Figure-level text
    # --------------------------------------------------------
    for txt in fig.texts:
        txt.set_fontsize(fs)


# ============================================================
# COLORBAR STYLING
# ============================================================

def style_colorbar(fig, fs=FS):

    # SHAP normally creates an additional axis for colorbar
    if len(fig.axes) > 1:

        cbar_ax = fig.axes[-1]

        cbar_ax.tick_params(
            axis="both",
            labelsize=fs
        )

        cbar_ax.set_ylabel(
            "Feature value",
            fontsize=fs
        )

        for txt in cbar_ax.get_yticklabels():
            txt.set_fontsize(fs)

        for txt in cbar_ax.get_xticklabels():
            txt.set_fontsize(fs)


# ============================================================
# SAVE FIGURE
# ============================================================

def save_figure(name):

    # --------------------------------------------------------
    # Vector PDF
    # --------------------------------------------------------
    plt.savefig(
        f"{name}.pdf",
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.05
    )

    # --------------------------------------------------------
    # PNG 300 dpi
    # --------------------------------------------------------
    plt.savefig(
        f"{name}.png",
        format="png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05
    )


# ============================================================
# ============================================================
# RANDOM FOREST SHAP
# ============================================================
# ============================================================

# ------------------------------------------------------------
# SHAP Explainer
# ------------------------------------------------------------

explainer_rf = shap.Explainer(
    best_rf,
    X_train
)

shap_values_rf = explainer_rf(X_test)

print("RF SHAP values shape :", shap_values_rf.values.shape)
print("RF base values shape :", shap_values_rf.base_values.shape)
print("RF data shape        :", shap_values_rf.data.shape)


# ============================================================
# RANDOM FOREST - POSITIVE CLASS
# ============================================================

# For binary classification with SHAP output:
# shape = (n_samples, n_features, n_classes)

if shap_values_rf.values.ndim == 3:

    shap_values_rf_class1 = shap_values_rf[:, :, 1]

else:

    shap_values_rf_class1 = shap_values_rf


# Apply publication-friendly feature names
shap_values_rf_class1.feature_names = NAMES


# ============================================================
# FIGURE 5
# SHAP BEESWARM - RANDOM FOREST
# ============================================================

plt.close("all")

shap.plots.beeswarm(
    shap_values_rf_class1,
    max_display=len(FEATURES),
    show=False,
    plot_size=(5.8, 4.1)
)

fig = plt.gcf()

# Explicit final figure size
fig.set_size_inches(5.8, 4.1)

# Apply manuscript font
style_fig(fig, FS)

# Style colorbar
style_colorbar(fig, FS)

# ------------------------------------------------------------
# No figure title
# Title should be provided in LaTeX caption
# ------------------------------------------------------------

fig.tight_layout(pad=0.6)

save_figure(
    "Figure_5_SHAP_RF_beeswarm"
)

plt.show()
plt.close(fig)


# ============================================================
# RANDOM FOREST - FEATURE IMPORTANCE
# ============================================================

plt.close("all")

shap.plots.bar(
    shap_values_rf_class1,
    max_display=len(FEATURES),
    show=False
)

fig = plt.gcf()

fig.set_size_inches(
    5.2,
    3.7
)

ax = plt.gca()

ax.set_xlabel(
    "Mean |SHAP value| (probability scale)",
    fontsize=FS
)

style_fig(fig, FS)

fig.tight_layout(pad=0.6)

save_figure(
    "Figure_SHAP_RF_bar"
)

plt.show()
plt.close(fig)


# ============================================================
# ============================================================
# XGBOOST SHAP
# ============================================================
# ============================================================

explainer_xgb = shap.TreeExplainer(
    best_xgb,
    feature_perturbation="tree_path_dependent"
)

shap_values_xgb = explainer_xgb(X_test)

print("XGB SHAP values shape :", shap_values_xgb.values.shape)
print("XGB base values shape :", shap_values_xgb.base_values.shape)
print("XGB data shape        :", shap_values_xgb.data.shape)


# Apply publication-friendly feature names
shap_values_xgb.feature_names = NAMES


# ============================================================
# FIGURE 6
# SHAP BEESWARM - XGBOOST
# ============================================================

plt.close("all")

shap.plots.beeswarm(
    shap_values_xgb,
    max_display=len(FEATURES),
    show=False,
    plot_size=(5.8, 4.1)
)

fig = plt.gcf()

fig.set_size_inches(
    5.8,
    4.1
)

style_fig(fig, FS)

style_colorbar(fig, FS)

fig.tight_layout(pad=0.6)

save_figure(
    "Figure_6_SHAP_XGB_beeswarm"
)

plt.show()
plt.close(fig)


# ============================================================
# XGBOOST - FEATURE IMPORTANCE
# ============================================================

plt.close("all")

shap.plots.bar(
    shap_values_xgb,
    max_display=len(FEATURES),
    show=False
)

fig = plt.gcf()

fig.set_size_inches(
    5.2,
    3.7
)

ax = plt.gca()

ax.set_xlabel(
    "Mean |SHAP value| (log-odds scale)",
    fontsize=FS
)

style_fig(fig, FS)

fig.tight_layout(pad=0.6)

save_figure(
    "Figure_SHAP_XGB_bar"
)

plt.show()
plt.close(fig)
# ============================================================
# FIGURE 7 — RANDOM FOREST WATERFALL
# ============================================================

plt.close("all")

fig = plt.figure(figsize=(5.2, 4.2))

shap.plots.waterfall(
    shap_values_rf_class1[0],
    max_display=8,
    show=False
)

ax = plt.gca()

# ============================================================
# FONT SIZE — DISESUAIKAN AGAR SETARA DENGAN FIGURE 6
# ============================================================

FEATURE_FS = 10
VALUE_FS   = 9
TICK_FS    = 9
LABEL_FS   = 10

# ============================================================
# STYLE FIGURE
# ============================================================

style_fig(fig, fs=FS)

# ============================================================
# ATUR SEMUA TEKS INTERNAL WATERFALL
# ============================================================

for txt in ax.texts:

    txt.set_fontsize(VALUE_FS)

# ============================================================
# PERBESAR NAMA FITUR
# ============================================================

for label in ax.get_yticklabels():
    label.set_fontsize(FEATURE_FS)

# ============================================================
# PERBESAR TICK SUMBU X
# ============================================================

for label in ax.get_xticklabels():
    label.set_fontsize(TICK_FS)

ax.tick_params(
    axis="x",
    labelsize=TICK_FS
)

# ============================================================
# LABEL SUMBU X
# ============================================================

ax.set_xlabel(
    "SHAP value (impact on model output)",
    fontsize=LABEL_FS
)

# ============================================================
# PINDAHKAN ANOTASI SHAP KECIL
# ============================================================

from matplotlib.transforms import ScaledTranslation

for txt in ax.texts:

    label = txt.get_text().strip()

    # --------------------------------------------------------
    # NILAI POSITIF KECIL
    # --------------------------------------------------------

    if label in [
        "+0.04",
        "+0.03",
        "+0.02",
        "+0.01"
    ]:

        txt.set_transform(
            txt.get_transform()
            + ScaledTranslation(
                10 / 72,
                0,
                fig.dpi_scale_trans
            )
        )

        txt.set_ha("left")
        txt.set_fontsize(VALUE_FS)
        txt.set_clip_on(False)

    # --------------------------------------------------------
    # NILAI NEGATIF KECIL
    # --------------------------------------------------------

    elif label in [
        "−0.04",
        "−0.03",
        "−0.02",
        "−0.01",
        "-0.04",
        "-0.03",
        "-0.02",
        "-0.01"
    ]:

        txt.set_transform(
            txt.get_transform()
            + ScaledTranslation(
                -10 / 72,
                0,
                fig.dpi_scale_trans
            )
        )

        txt.set_ha("right")
        txt.set_fontsize(VALUE_FS)
        txt.set_clip_on(False)


# ============================================================
# LAYOUT
# ============================================================

fig.tight_layout(pad=0.8)

# ============================================================
# SIMPAN FIGURE 7
# ============================================================

save_figure("Fig7_RF_waterfall")

plt.show()
plt.close(fig)


# ============================================================
# FIGURE 8 — XGBOOST WATERFALL
# ============================================================

plt.close("all")

fig = plt.figure(figsize=(5.2, 4.2))

shap.plots.waterfall(
    shap_values_xgb[0],
    max_display=8,
    show=False
)

ax = plt.gca()

# ============================================================
# FONT SIZE — DISESUAIKAN AGAR SETARA DENGAN FIGURE 6
# ============================================================

FEATURE_FS = 10
VALUE_FS   = 9
TICK_FS    = 9
LABEL_FS   = 10

# ============================================================
# STYLE FIGURE
# ============================================================

style_fig(fig, fs=FS)

# ============================================================
# ATUR SEMUA TEKS INTERNAL WATERFALL
# ============================================================

for txt in ax.texts:
    txt.set_fontsize(VALUE_FS)

# ============================================================
# PERBESAR NAMA FITUR
# ============================================================

for label in ax.get_yticklabels():
    label.set_fontsize(FEATURE_FS)

# ============================================================
# PERBESAR TICK SUMBU X
# ============================================================

for label in ax.get_xticklabels():
    label.set_fontsize(TICK_FS)

ax.tick_params(
    axis="x",
    labelsize=TICK_FS
)

# ============================================================
# LABEL SUMBU X
# ============================================================

ax.set_xlabel(
    "SHAP value (impact on model output)",
    fontsize=LABEL_FS
)

# ============================================================
# PINDAHKAN ANOTASI SHAP KECIL
# ============================================================

from matplotlib.transforms import ScaledTranslation

for txt in ax.texts:

    label = txt.get_text().strip()

    # --------------------------------------------------------
    # NILAI POSITIF KECIL
    # --------------------------------------------------------

    if label in [
        "+0.04",
        "+0.03",
        "+0.02",
        "+0.01"
    ]:

        txt.set_transform(
            txt.get_transform()
            + ScaledTranslation(
                10 / 72,
                0,
                fig.dpi_scale_trans
            )
        )

        txt.set_ha("left")
        txt.set_fontsize(VALUE_FS)
        txt.set_clip_on(False)

    # --------------------------------------------------------
    # NILAI NEGATIF KECIL
    # --------------------------------------------------------

    elif label in [
        "−0.04",
        "−0.03",
        "−0.02",
        "−0.01",
        "-0.04",
        "-0.03",
        "-0.02",
        "-0.01"
    ]:

        txt.set_transform(
            txt.get_transform()
            + ScaledTranslation(
                -10 / 72,
                0,
                fig.dpi_scale_trans
            )
        )

        txt.set_ha("right")
        txt.set_fontsize(VALUE_FS)
        txt.set_clip_on(False)


# ============================================================
# LAYOUT
# ============================================================

fig.tight_layout(pad=0.8)

# ============================================================
# SIMPAN FIGURE 8
# ============================================================

save_figure("Fig8_XGB_waterfall")

plt.show()
plt.close(fig)

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
    "shap_cross_model_comparison.csv",
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

# ============================================================
# C. BOOTSTRAP ROBUSTNESS OF CROSS-MODEL SHAP RANKING
# ============================================================

print("\n" + "=" * 70)
print("C. BOOTSTRAP ROBUSTNESS OF CROSS-MODEL SHAP RANKING")
print("=" * 70)


# ------------------------------------------------------------
# 1. Bootstrap settings
# ------------------------------------------------------------

N_BOOTSTRAP_SHAP = 2000
RANDOM_STATE_SHAP = 42

rng_shap = np.random.RandomState(
    RANDOM_STATE_SHAP
)


# ------------------------------------------------------------
# 2. Extract SHAP values
# ------------------------------------------------------------

rf_shap_matrix = np.asarray(
    shap_values_rf_class1.values
)

xgb_shap_matrix = np.asarray(
    shap_values_xgb_class1.values
)

features_shap = X_test.columns.tolist()

n_test = rf_shap_matrix.shape[0]


# ------------------------------------------------------------
# 3. Storage
# ------------------------------------------------------------

bootstrap_spearman = []

bootstrap_rf_ranks = []

bootstrap_xgb_ranks = []

bootstrap_rf_top3 = []

bootstrap_xgb_top3 = []

bootstrap_top3_agreement = []


# ------------------------------------------------------------
# 4. Bootstrap resampling
# ------------------------------------------------------------

for b in range(N_BOOTSTRAP_SHAP):

    # Same bootstrap sample for both models
    indices = rng_shap.randint(
        0,
        n_test,
        size=n_test
    )

    rf_boot = rf_shap_matrix[
        indices,
        :
    ]

    xgb_boot = xgb_shap_matrix[
        indices,
        :
    ]


    # --------------------------------------------------------
    # Mean absolute SHAP
    # --------------------------------------------------------

    rf_mean_abs = np.abs(
        rf_boot
    ).mean(axis=0)

    xgb_mean_abs = np.abs(
        xgb_boot
    ).mean(axis=0
    )


    # --------------------------------------------------------
    # Rank features
    # --------------------------------------------------------

    rf_rank = pd.Series(
        rf_mean_abs,
        index=features_shap
    ).rank(
        ascending=False,
        method="min"
    )

    xgb_rank = pd.Series(
        xgb_mean_abs,
        index=features_shap
    ).rank(
        ascending=False,
        method="min"
    )


    # --------------------------------------------------------
    # Spearman correlation
    # --------------------------------------------------------

    rho_boot, _ = spearmanr(
        rf_rank.values,
        xgb_rank.values
    )

    bootstrap_spearman.append(
        rho_boot
    )


    # --------------------------------------------------------
    # Store rankings
    # --------------------------------------------------------

    bootstrap_rf_ranks.append(
        rf_rank.values
    )

    bootstrap_xgb_ranks.append(
        xgb_rank.values
    )


    # --------------------------------------------------------
    # Top-3 features
    # --------------------------------------------------------

    rf_top3 = set(
        np.argsort(
            rf_mean_abs
        )[-3:]
    )

    xgb_top3 = set(
        np.argsort(
            xgb_mean_abs
        )[-3:]
    )


    bootstrap_rf_top3.append(
        rf_top3
    )

    bootstrap_xgb_top3.append(
        xgb_top3
    )


    # --------------------------------------------------------
    # Top-3 agreement
    # --------------------------------------------------------

    bootstrap_top3_agreement.append(
        rf_top3 == xgb_top3
    )


# ------------------------------------------------------------
# 5. Bootstrap Spearman results
# ------------------------------------------------------------

bootstrap_spearman = np.asarray(
    bootstrap_spearman
)

rho_median = np.median(
    bootstrap_spearman
)

rho_lower = np.percentile(
    bootstrap_spearman,
    2.5
)

rho_upper = np.percentile(
    bootstrap_spearman,
    97.5
)


print("\nBootstrap Spearman agreement:")

print(
    f"Median rho = {rho_median:.4f}"
)

print(
    f"95% CI = "
    f"({rho_lower:.4f}, {rho_upper:.4f})"
)


# ------------------------------------------------------------
# 6. Bootstrap Top-3 agreement
# ------------------------------------------------------------

top3_agreement_rate = np.mean(
    bootstrap_top3_agreement
)

print("\nBootstrap Top-3 agreement:")

print(
    f"Agreement = "
    f"{top3_agreement_rate * 100:.2f}%"
)


# ------------------------------------------------------------
# 7. Feature-level rank stability
# ------------------------------------------------------------

bootstrap_rf_ranks = np.asarray(
    bootstrap_rf_ranks
)

bootstrap_xgb_ranks = np.asarray(
    bootstrap_xgb_ranks
)


rank_stability = pd.DataFrame({

    "Feature": features_shap,

    "RF_Median_Rank": np.median(
        bootstrap_rf_ranks,
        axis=0
    ),

    "RF_Rank_CI_Lower": np.percentile(
        bootstrap_rf_ranks,
        2.5,
        axis=0
    ),

    "RF_Rank_CI_Upper": np.percentile(
        bootstrap_rf_ranks,
        97.5,
        axis=0
    ),

    "XGB_Median_Rank": np.median(
        bootstrap_xgb_ranks,
        axis=0
    ),

    "XGB_Rank_CI_Lower": np.percentile(
        bootstrap_xgb_ranks,
        2.5,
        axis=0
    ),

    "XGB_Rank_CI_Upper": np.percentile(
        bootstrap_xgb_ranks,
        97.5,
        axis=0
    )
})


# ------------------------------------------------------------
# 8. Top-3 frequency of each feature
# ------------------------------------------------------------

rf_top3_frequency = []

xgb_top3_frequency = []


for feature_index in range(
    len(features_shap)
):

    rf_count = sum(
        feature_index in top3
        for top3 in bootstrap_rf_top3
    )

    xgb_count = sum(
        feature_index in top3
        for top3 in bootstrap_xgb_top3
    )

    rf_top3_frequency.append(
        rf_count / N_BOOTSTRAP_SHAP
    )

    xgb_top3_frequency.append(
        xgb_count / N_BOOTSTRAP_SHAP
    )


rank_stability[
    "RF_Top3_Frequency"
] = rf_top3_frequency

rank_stability[
    "XGB_Top3_Frequency"
] = xgb_top3_frequency


# ------------------------------------------------------------
# 9. Sort by RF median rank
# ------------------------------------------------------------

rank_stability = rank_stability.sort_values(
    "RF_Median_Rank"
).reset_index(
    drop=True
)


# ------------------------------------------------------------
# 10. Display results
# ------------------------------------------------------------

print(
    "\nFeature-level SHAP rank stability:"
)

print(
    rank_stability.round(3)
)


# ------------------------------------------------------------
# 11. Save results
# ------------------------------------------------------------

rank_stability.to_csv(
    "shap_bootstrap_rank_stability.csv",
    index=False
)

pd.DataFrame({
    "Bootstrap_Spearman_Rho":
        bootstrap_spearman
}).to_csv(
    "shap_bootstrap_spearman.csv",
    index=False
)


print(
    "\nBootstrap SHAP results saved:"
)

print(
    "shap_bootstrap_rank_stability.csv"
)

print(
    "shap_bootstrap_spearman.csv"
)

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
    "bootstrap_95CI_results.csv",
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
    "delong_auc_comparison.csv",
    index=False
)

print(
    "\nHasil DeLong disimpan sebagai: "
    "delong_auc_comparison.csv"
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
    "spearman_anthropometric_correlation.csv"
)

print(
    "\nCorrelation matrix disimpan sebagai: "
    "spearman_anthropometric_correlation.csv"
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