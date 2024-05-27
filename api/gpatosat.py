from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource
from model.gpatosats import GPAtoSATModel
from flask_cors import CORS

gpatosat_api = Blueprint('gpatosat_api', __name__, url_prefix='/api/gpatosat')
api = Api(gpatosat_api)
# CORS(satgpacross_api)  # Enable CORS for the API

class GpatosatAPI:
    class _CRUD(Resource):
        def post(self):
            print('i think it worked')
            data = request.get_json()
            gpa = data.get('gpa')
            try:
                # Attempt to convert the values to integers
                # gpa = int(gpa)
                gpa = float(gpa)
            except (TypeError, ValueError):
                return jsonify({"error": "Invalid data format"}), 400

            # Instantiate the SATtoGPAModel class with validated values
            model = GPAtoSATModel(gpa=gpa)
            prediction = model.predict()
            print(prediction)
            # Return the prediction as JSON

            return jsonify({"prediction": str(prediction)})

# Add the CRUD resource to the API
api.add_resource(GpatosatAPI._CRUD, '/')