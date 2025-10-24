from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import json

app = Flask(__name__)

# Load the trained model
model_path = "model/house_price_model.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Load feature ranges
ranges_path = "model/feature_ranges.json"
with open(ranges_path, "r") as file:
    feature_ranges = json.load(file)

# Route to render the prediction form
@app.route("/")
def home():
    return render_template("predict.html", ranges=feature_ranges)

# Route to handle prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get user inputs
        inputs = [
            float(request.form.get("OverallQual")),
            float(request.form.get("GrLivArea")),
            float(request.form.get("GarageCars")),
            float(request.form.get("GarageArea")),
            float(request.form.get("TotalBsmtSF")),
            float(request.form.get("1stFlrSF")),
            float(request.form.get("FullBath")),
            float(request.form.get("TotRmsAbvGrd"))
        ]

        # Prepare input for prediction
        input_array = np.array([inputs]).reshape(1, -1)

        # Make prediction
        predicted_price = model.predict(input_array)[0]

        # Return the result
        return jsonify({
            "input": {
                "OverallQual": inputs[0],
                "GrLivArea": inputs[1],
                "GarageCars": inputs[2],
                "GarageArea": inputs[3],
                "TotalBsmtSF": inputs[4],
                "1stFlrSF": inputs[5],
                "FullBath": inputs[6],
                "TotRmsAbvGrd": inputs[7]
            },
            "predicted_price": round(predicted_price, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
