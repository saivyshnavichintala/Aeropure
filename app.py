from flask import Flask, render_template

from Load_Data import load_data
from EDA import run_eda
from Preprocessing import preprocess_data
from Regression import run_regression
from Decision_Tree import run_decision_tree


app = Flask(__name__)


# Load dataset
df = load_data()


# =====================================================
# HOME
# =====================================================

@app.route("/")
def home():
    return render_template("home.html")


# =====================================================
# DATA
# =====================================================

@app.route("/data")
def data_page():

    data = df.head(100).to_html(
        classes="data-table",
        index=False
    )

    return render_template(
        "data.html",
        table=data,
        rows=len(df),
        columns=len(df.columns)
    )


# =====================================================
# EDA
# =====================================================

@app.route("/eda")
def eda_page():

    results = run_eda(df)

    return render_template(
        "eda.html",
        results=results
    )


# =====================================================
# PREPROCESSING
# =====================================================

@app.route("/preprocessing")
def preprocessing_page():

    results = preprocess_data(df)

    return render_template(
        "preprocessing.html",
        results=results
    )


# =====================================================
# REGRESSION
# =====================================================

@app.route("/regression")
def regression_page():

    results = run_regression(df)

    return render_template(
        "regression.html",
        results=results
    )


# =====================================================
# DECISION TREE
# =====================================================

@app.route("/decision-tree")
def decision_tree_page():

    results = run_decision_tree(df)

    return render_template(
        "decision_tree.html",
        results=results
    )


# =====================================================
# RUN APPLICATION
# =====================================================

if __name__ == "__main__":
    app.run(debug=True)