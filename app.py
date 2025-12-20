from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import os

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

# Urutan kolom fitur sesuai dataset
FEATURE_ORDER = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

CATEGORICAL_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_PATH)

missing_cols = [c for c in FEATURE_ORDER + [TARGET_COL] if c not in df.columns]
if missing_cols:
    raise ValueError(f"Kolom berikut tidak ditemukan di heart.csv: {missing_cols}")

X = df[FEATURE_ORDER].astype(float).to_numpy()
y = df[TARGET_COL].astype(int).to_numpy()

# =========================
# DROPDOWN DATA
# =========================
def build_dropdown_values(df: pd.DataFrame) -> dict:
    return {
        col: sorted(df[col].dropna().unique().tolist())
        for col in CATEGORICAL_COLS
    }

# =========================
# STANDARD SCALER (ringan)
# =========================
class SimpleStandardScaler:
    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

# =========================
# GAUSSIAN NAIVE BAYES (ringan)
# =========================
class SimpleGaussianNB:
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.priors_ = {}
        self.means_ = {}
        self.vars_ = {}

        for c in self.classes_:
            Xc = X[y == c]
            self.priors_[c] = Xc.shape[0] / X.shape[0]
            self.means_[c] = Xc.mean(axis=0)
            v = Xc.var(axis=0)
            v[v == 0] = 1e-9
            self.vars_[c] = v
        return self

    def _log_prob(self, X, mean, var):
        return -0.5 * np.sum(np.log(2 * np.pi * var) + ((X - mean) ** 2) / var, axis=1)

    def predict_proba(self, X):
        log_probs = []
        for c in self.classes_:
            lp = np.log(self.priors_[c]) + self._log_prob(X, self.means_[c], self.vars_[c])
            log_probs.append(lp)
        log_probs = np.vstack(log_probs).T
        log_probs -= log_probs.max(axis=1, keepdims=True)
        probs = np.exp(log_probs)
        return probs / probs.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

# =========================
# TRAIN MODEL (sekali saat start)
# =========================
scaler = SimpleStandardScaler().fit(X)
X_scaled = scaler.transform(X)

model = SimpleGaussianNB().fit(X_scaled, y)

# =========================
# ROUTES
# =========================
@app.route("/")
def home():
    return render_template(
        "index.html",
        data=build_dropdown_values(df),
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

        for col in FEATURE_ORDER:
            val = request.form.get(col, "").strip()
            if val == "":
                raise ValueError(f"Kolom '{col}' belum diisi.")
            filled[col] = val
            features.append(float(val))

        X_input = np.array([features])
        X_input_scaled = scaler.transform(X_input)

        pred = int(model.predict(X_input_scaled)[0])
        proba = model.predict_proba(X_input_scaled)[0]
        conf = float(np.max(proba) * 100)

        result = (
            "💔 Pasien <b>TERINDIKASI</b> mengidap penyakit jantung (target=1) 💔"
            if pred == 1 else
            "💖 Pasien <b>TIDAK</b> mengidap penyakit jantung (target=0) 💖"
        )

        proba_text = (
            f"Probabilitas: P(0)={proba[0]:.4f}, "
            f"P(1)={proba[1]:.4f} | "
            f"Keyakinan: {conf:.2f}%"
        )

        return render_template(
            "index.html",
            prediction_text=result,
            proba_text=proba_text,
            data=dropdown_data,
            feature_cols=FEATURE_ORDER,
            filled=filled,
            labels=["Tidak (0)", "Ya (1)"],
            values=[float(proba[0]), float(proba[1])]
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
