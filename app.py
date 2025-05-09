from flask import Flask, render_template, request, jsonify
import pandas as pd
import os

app = Flask(__name__)
excel_file = "admission_data.xlsx"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/submit_form", methods=["POST"])
def submit_form():
    data = request.get_json()
    df = pd.DataFrame([data])
    if os.path.exists(excel_file):
        df.to_excel(excel_file, index=False, mode='a', header=False)
    else:
        df.to_excel(excel_file, index=False)
    return jsonify({"message": "Form submitted and saved to Excel!"})

if __name__ == "__main__":
    app.run(debug=True)