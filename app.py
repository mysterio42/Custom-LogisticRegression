from flask import Flask, jsonify, request
from flask_restful import Resource, Api

from utils.model import latest_modified_weight, load_model

app = Flask(__name__)
api = Api(app)

model = load_model(latest_modified_weight())


class BinaryClassifier(Resource):

    def post(self):
        posted_data = request.get_json()
        if 'scalar' in posted_data:
            scalar = posted_data['scalar']
            ans = model.predict([[float(scalar)]])[0]
            return jsonify({'prediction': {'class': str(ans)}})
        else:
            return jsonify({'message': 'provide scalar value to predict'})


api.add_resource(BinaryClassifier, '/classify')

if __name__ == '__main__':
    app.run(host='0.0.0.0')
