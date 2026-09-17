from flask import Flask, render_template, request

from Load_Data import load_data
from EDA import run_eda
from preprocess import run_preprocessing
from linear_regression import run_regression_analysis
from logistic_regression import run_logistic_regression_analysis
from decision_tree import run_decision_tree_analysis


app = Flask(__name__)


# ============================================================
# LOAD DATA
# ============================================================

df = load_data()


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# DATA
# ============================================================

@app.route("/data")
def data_page():

    summary = {

        "n_rows":
            len(df),

        "n_cols":
            len(df.columns),

        "columns":
            list(df.columns),

        "dtypes":
            {
                col: str(dtype)
                for col, dtype
                in df.dtypes.items()
            },

        "missing_counts":
            {
                col:
                    int(
                        df[col].isna().sum()
                    )
                for col in df.columns
            },

        "preview":
            df.head(
                10
            ).to_dict(
                orient="records"
            )
    }

    return render_template(
        "data.html",
        summary=summary
    )


# ============================================================
# EDA
# ============================================================

@app.route("/eda")
def eda_page():

    # run_eda() does not take df as an argument
    results = run_eda()

    return render_template(
        "eda.html",
        result=results
    )


# ============================================================
# PREPROCESSING
# ============================================================

@app.route("/preprocessing")
def preprocessing_page():

    # run_preprocessing() does not take df as an argument
    results = run_preprocessing()

    return render_template(
        "preprocessing.html",
        result=results
    )


# ============================================================
# LINEAR REGRESSION
# ============================================================

@app.route("/linear-regression")
def linear_regression_page():

    model_type = request.args.get(
        "model",
        "linear"
    ).lower()

    allowed_models = [
        "linear",
        "ridge",
        "lasso"
    ]

    if model_type not in allowed_models:

        model_type = "linear"

    results = run_regression_analysis(
        model_type=model_type
    )

    return render_template(
        "linear_regression.html",
        result=results
    )


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

@app.route("/logistic-regression")
def logistic_regression_page():

    model_type = request.args.get(
        "model",
        "logistic"
    ).lower()

    allowed_models = [
        "logistic",
        "ridge",
        "lasso"
    ]

    if model_type not in allowed_models:

        model_type = "logistic"

    results = (
        run_logistic_regression_analysis(
            model_type=model_type
        )
    )

    return render_template(
        "logistic_regression.html",
        result=results
    )


# ============================================================
# DECISION TREE
# ============================================================

@app.route("/decision-tree")
def decision_tree_page():

    model_type = request.args.get(
        "model",
        "cart"
    ).lower()

    allowed_models = [

        "id3",

        "c45",

        "cart",

        "pruned",

        "random_forest",

        "adaboost",

        "gradient_boosting",

        "xgboost",

        "lightgbm"
    ]

    if model_type not in allowed_models:

        model_type = "cart"

    result = run_decision_tree_analysis(
        model_type=model_type
    )

    return render_template(
        "decision_tree.html",
        result=result
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )