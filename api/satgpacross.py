from flask import Blueprint, request, jsonify, Flask
from flask_restful import Api, Resource # used for REST API building
from model.satgpacrosses import SATtoGPAModel
from flask_cors import CORS

satgpacross_api = Blueprint('satgpacross_api', __name__,
                   url_prefix='/satgpagross')

# API docs https://flask-restful.readthedocs.io/en/latest/api.html
api = Api(satgpacross_api)

class SatgpacrossAPI:        
    class _CRUD(Resource):  # User API operation for Create, Read.  THe Update, Delete methods need to be implemeented
        def post(self): # Create method            
            data = request.get_json()
            SATtoGPAModel = SATtoGPAModel(data.get('satscore'), data.get('GPA'))
            return SATtoGPAModel.predict()

            
    # building RESTapi endpoint
    api.add_resource(_CRUD, '/')