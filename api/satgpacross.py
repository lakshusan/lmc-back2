from flask import Blueprint, request, jsonify, Flask
from flask_restful import Api, Resource # used for REST API building
from model.satgpacrosses import SATtoGPAModel
from flask_cors import CORS

# Initialize ExerciseModel instance and train the models
satscore_model = SATtoGPAModel.get_instance()
satscore_model.init_satscore_list()
satscore_model._clean()
satscore_model._train()  # Call _train method to train the models

satgpacross_api = Blueprint('satgpacross_api', __name__,
                   url_prefix='/satgpagross')

# API docs https://flask-restful.readthedocs.io/en/latest/api.html
api = Api(satgpacross_api)

class SatgpacrossAPI:        
    class _CRUD(Resource):  # User API operation for Create, Read.  THe Update, Delete methods need to be implemeented
        def post(self): # Create method            
            print("Request received at /satgpacross endpoint")  # Debugging
            data = request.get_json()
            print("Received data:", data)  # Debugging
            sat_score = data['SAT_score']
            response = satscore_model.predict({'time': sat_score})
            print("Response:", response)  # Debugging
            return jsonify(response)

            
    # building RESTapi endpoint
    api.add_resource(_CRUD, '/')