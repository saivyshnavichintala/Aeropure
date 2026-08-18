from flask import Flask, render_template

from Load_Data import load_data
from EDA import run_eda
from preprocess import run_preprocessing

app = Flask(__name__)


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# DATA
# =========================================================

@app.route("/data")
def data_page():

    try:

        data = load_data()

        summary = {

            "n_rows": len(data),

            "n_cols": len(data.columns),

            "columns": data.columns.tolist(),

            "dtypes":
                data.dtypes.astype(str).to_dict(),

            "missing_counts":
                data.isnull().sum().to_dict(),

            "preview":
                data.head(10).to_dict(
                    orient="records"
                )
        }

        return render_template(
            "data.html",
            summary=summary,
            error=None
        )

    except Exception as e:

        return render_template(
            "data.html",
            summary=None,
            error=str(e)
        )


# =========================================================
# EDA
# =========================================================

@app.route("/eda")
def eda_page():

    try:

        result = run_eda()

        return render_template(
            "eda.html",
            result=result,
            error=None
        )

    except Exception as e:

        return render_template(
            "eda.html",
            result=None,
            error=str(e)
        )


# =========================================================
# PREPROCESSING
# =========================================================

@app.route("/preprocessing")
def preprocessing_page():

    try:

        result = run_preprocessing()

        return render_template(
            "preprocessing.html",
            result=result,
            error=None
        )

    except Exception as e:

        return render_template(
            "preprocessing.html",
            result=None,
            error=str(e)
        )


# =========================================================
# FORECAST
# =========================================================

@app.route("/forecast")
def forecast_page():

    return render_template(
        "forecast.html"
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )