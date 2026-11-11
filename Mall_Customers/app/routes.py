from flask import render_template, request
from app import app
from app.model import predict_segment


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    if request.method == "POST":
        try:
            age = float(request.form["age"])
            income = float(request.form["income"])
            score = float(request.form["score"])
            if not (18 <= age <= 100 and 0 <= income <= 500 and 1 <= score <= 100):
                error = "Please enter Age 18-100, Income 0-500 (k$) and Score 1-100."
            else:
                result = predict_segment(age, income, score)
        except ValueError:
            error = "Please enter valid numbers."
    return render_template("index.html", result=result, error=error)
