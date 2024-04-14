from flask import Flask, render_template, request
import os
from reportlab.pdfgen import canvas

from utils.db_helpers import create_patient_record, query_patient_record
from utils.ai_utils import *

app = Flask(__name__)


# Function to get folder names in a specified directory
def get_folder_names(directory):
    return [
        folder
        for folder in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, folder))
    ]


# Function to diagnose based on folder name and symptoms
def diagnose(folder_name, symptoms):
    res = query_patient_record(folder_name)
    res = " ".join(str(res) for res in res)

    prompt = get_recommendation_prompt(res, "", symptoms)

    disgonosis = get_response(prompt)

    return disgonosis.content[0].text


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/rec", methods=["GET", "POST"])
def rec():
    directory = "/home/sharon/workspace/wow-hack/chromadb"
    folders = get_folder_names(directory)
    diagnosis_result = None

    if request.method == "POST":
        selected_folder = request.form["folder"]
        symptoms = request.form["symptoms"]
        diagnosis_result = diagnose(selected_folder, symptoms)

    return render_template(
        "rec.html", folders=folders, diagnosis_result=diagnosis_result
    )


def check_for_ade(selected_folder, drug_name):
    res = query_patient_record(selected_folder)
    res = " ".join(str(res) for res in res)

    prompt = get_ade_prompt(drug_name, res)

    ade = get_response(prompt)

    return ade.content[0].text


@app.route("/ade", methods=["GET", "POST"])
def ade():
    directory = "/home/sharon/workspace/wow-hack/chromadb"
    folders = get_folder_names(directory)
    ade_result = None

    if request.method == "POST":
        selected_folder = request.form["folder"]
        drug_name = request.form["drug_name"]
        ade_result = check_for_ade(selected_folder, drug_name)

    return render_template("ade.html", folders=folders, ade_result=ade_result)


def document_summarization(selected_folder):
    res = query_patient_record(selected_folder)
    res = " ".join(str(res) for res in res)

    prompt = get_summarization_prompt(res)

    summary = get_response(prompt)

    return summary.content[0].text


@app.route("/summary", methods=["GET", "POST"])
def summary():
    directory = "/home/sharon/workspace/wow-hack/chromadb"  # Specify the path to your documents directory
    folders = get_folder_names(directory)
    summary_result = None

    if request.method == "POST":
        selected_folder = request.form["folder"]
        summary_result = document_summarization(selected_folder)

    return render_template(
        "summary.html", folders=folders, summary_result=summary_result
    )


@app.route("/add", methods=["GET", "POST"])
def add_patient():
    if request.method == "POST":
        name = request.form["name"]
        age = request.form["age"]
        notes = request.form["notes"]
        drugs = request.form["drugs"]
        file_path = f"./patient_records/patient_{name}.pdf"
        c = canvas.Canvas(file_path)
        c.setFont("Helvetica", 24)
        c.drawString(100, 700, "Patient Record")
        c.setFont("Helvetica", 14)
        c.drawString(100, 650, f"Name: {name}")
        c.drawString(100, 625, f"Age: {age}")
        c.drawString(100, 600, f"Notes: {notes}")
        c.drawString(100, 575, f"Drugs: {drugs}")
        c.save()
        create_patient_record(name, file_path)
    return render_template("add_patient.html")


if __name__ == "__main__":
    app.run(debug=True)
