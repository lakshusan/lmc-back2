import pandas as pd
from sklearn.ensemble import RandomForestRegressor


class SATtoGPAModel:
    def __init__(self, satscore):
        self._satscore = satscore
        # self.target = 'Grade Point Average'

    def predict(self):
        df = pd.read_csv('sattogpa.csv')
        # Ensure the same number of features as the model expects
        X = df['SAT Score']  # Assuming 'SAT Score' is the only feature
        y = df['Grade Point Average']
        regressor = RandomForestRegressor(n_estimators=10, random_state=42)
        regressor.fit(X.values.reshape(-1, 1), y)  # Reshape X to be a 2D array
        # Predicting the GPA
        predicted_gpa = regressor.predict([[self._satscore]])[0]
        return predicted_gpa

# # Flask API
# from flask import Flask, request, jsonify

# app = Flask(__name__)
# model = SATtoGPAModel()

# @app.route('/satgpacross', methods=['POST'])
# def predict_gpa():
#     data = request.get_json()
#     satscore = data.get('satscore')
#     if satscore is None:
#         return jsonify({"error": "Missing SAT score"}), 400

#     # Make prediction using the SATtoGPAModel
#     gpa_prediction = model.predict(satscore)

#     # Return the prediction as JSON
#     return jsonify({"GPA_estimate": gpa_prediction})

# if __name__ == '__main__':
#     app.run(debug=True)