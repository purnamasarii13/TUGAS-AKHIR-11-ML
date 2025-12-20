from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import os

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

# =========================
# FLASK APP
# =========================
app = Flask(__name__)

# =========================
# PATH AMAN (UNTUK VERCEL)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "heart.csv")

TARGET_COL = "target"

# Urutan kolom fitur sesuai dataset heart.csv
FEATURE_ORDER = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

# Kolom kategori (sudah numerik, tapi ditampilkan sebagai dropdown)
CATEGORICAL_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_PATH)

# Validasi kolom wajib
missing_cols = [c for c in FEATURE_ORDER + [TARGET_COL] if c not in df.columns]
if missing_cols:
    raise ValueError(f"Kolom berikut tidak ditemukan di heart.csv: {missing_cols}")

X = df[FEATURE_ORDER]
y = df[TARGET_COL].astype(int)


# =========================
# DROPDOWN DATA
# =========================
def build_dropdown_values(df: pd.DataFrame) -> dict:
    data = {}
    for col in CATEGORICAL_COLS:
        data[col] = sorted(df[col].dropna().unique().tolist())
    return data


# =========================
# TRAIN MODEL (SEKALI SAAT START)
# =========================
preprocess = ColumnTransformer(
    transformers=[("num", StandardScaler(), FEATURE_ORDER)],
    remainder="drop"
)

model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("clf", GaussianNB())
])

# Latih model pakai seluruh data
# (Evaluasi sudah dilakukan di ipynb)
model.fit(X, y)


# =========================
# ROUTES
# =========================
@app.route("/")
def home():
    dropdown_data = build_dropdown_values(df)
    return render_template(
        "index.html",
        data=dropdown_data,
        feature_cols=FEATURE_ORDER,
        filled=None,
        labels=None,
        values=None
    )


@app.route("/predict", methods=["POST"])
def predict():
    dropdown_data = build_dropdown_values(df)

    try:
        filled = {}
        features = []

        # Ambil input sesuai urutan dataset
        for col in FEATURE_ORDER:
            raw_val = request.form.get(col, "").strip()
            filled[col] = raw_val

            if raw_val == "":
                raise ValueError(f"Kolom '{col}' belum diisi.")

            features.append(float(raw_val))

        X_input = pd.DataFrame([features], columns=FEATURE_ORDER)

        # Prediksi
        pred_class = int(model.predict(X_input)[0])
        proba = model.predict_proba(X_input)[0]  # [P(0), P(1)]
        conf_pct = float(np.max(proba) * 100)

        # Hasil teks
        result = (
            "💔 Pasien <b>TERINDIKASI</b> mengidap penyakit jantung (target=1) 💔"
            if pred_class == 1 else
            "💖 Pasien <b>TIDAK</b> mengidap penyakit jantung (target=0) 💖"
        )

        proba_text = (
            f"Probabilitas: P(0)={proba[0]:.4f}, "
            f"P(1)={proba[1]:.4f} | "
            f"Keyakinan: {conf_pct:.2f}%"
        )

        # Data grafik (Plotly)
        labels = ["Tidak (0)", "Ya (1)"]
        values = [float(proba[0]), float(proba[1])]

        return render_template(
            "index.html",
            prediction_text=result,
            proba_text=proba_text,
            data=dropdown_data,
            feature_cols=FEATURE_ORDER,
            filled=filled,
            labels=labels,
            values=values
        )

    except Exception as e:
        return render_template(
            "index.html",
            error_text=f"Terjadi error: {e}",
            data=dropdown_data,
            feature_cols=FEATURE_ORDER,
            filled=None,
            labels=None,
            values=None
        )


# =========================
# RUN LOCAL
# =========================
if __name__ == "__main__":
    app.run(debug=True)
