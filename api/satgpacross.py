from flask import Blueprint, request, jsonify, Flask
from flask_restful import Api, Resource # used for REST API building
from model.satgpacrosses import SATtoGPAModel
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
satgpacross_api = Blueprint('satgpacross_api', __name__,
                   url_prefix='/api/satgpacross')

# API docs https://flask-restful.readthedocs.io/en/latest/api.html
api = Api(satgpacross_api)

# Initialize ExerciseModel instance and train the models
satscore_model = SATtoGPAModel.get_instance()
satscore_model.init_satscore_list()
satscore_model._clean()
satscore_model._train()  # Call _train method to train the models

@app.route('/predict', methods=['POST'])
def post():
    print("Request received at /predict endpoint")  # Debugging
    data = request.get_json()
    print("Received data:", data)  # Debugging
    sat_score = data['SAT_score']
    response = satscore_model.predict({'time': sat_score})
    print("Response:", response)  # Debugging
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)