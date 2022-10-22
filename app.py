import numpy as np
from flask import Flask, jsonify, request, render_template

import pickle

app = Flask(__name__)
model = pickle.load(open("modelo.pkl", "rb"))
names = {0: "Inaceitável", 1: "Aceitável", 2: "Bom", 3: "Muito Bom"}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    pred = model.predict(final_features)

    output = names[pred[0]]

    return render_template("index.html", prediction_text="Aceitabilidade do Veículo: " + output)


@app.route("/api", methods=["POST"])
def results():
    data = request.get_json(force=True)
    pred = model.predict([np.array(list(data.values()))])

    output = names[pred[0]]
    return jsonify(output)
