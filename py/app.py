from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import numpy as np 
import json


from start_server import ModelServer
import normals_model3 as nm 


app = Flask(__name__, static_folder='.')

CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "expose_headers": ["Content-Type"],
        "supports_credentials": False,
        "max_age": 3600
    }
})



model_instances = {}

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)

app.json_encoder = NumpyEncoder



@app.route("/")
def index():
    return send_from_directory("static", 'index.html')


@app.route('/api/model/create', methods=['POST'])
def create_model():
    try:
        data = request.json
        
        model = nm.Model() 
        center = data.get('center', [0, 0,0])
        model.update_start(*center, 0)

        x_func = data.get('x_func', 'x')
        y_func = data.get('y_func', 'y')
        model.update_function(x=x_func, y=y_func)


        if 'constants' in data:
            model.update_constants(**data['constants'])
        if 'epsilon' in data: 
            model.epsilon = data['epsilon']
        if 'density' in data: 
            model.point_density = data['density']

        model.first_step(three_dimensional=False)

        return jsonify({
            'success': True,
            'points': model.points.tolist(),
            'normals': model.normals.tolist(),
            'step': 0,
            'config': {
                'epsilon': model.epsilon,
                'density': model.point_density
            }
        }) 
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    




@app.route("/api/health", methods=['GET'])
def health():
    return jsonify({'success': True, 'status': 'healthy'})


@app.route('/api/compute/initial', methods=['POST'])
def compute_initial():
    try:
        data = request.json

        model = nm.Model()
        center = data.get('center', [0, 0, 0])
        if len(center) == 2:
            center = center + [0]
        model.update_start(*center)

        x_func = data.get('x_func', 'x')
        y_func = data.get('y_func', 'y')
        model.update_function(x=x_func, y=y_func)

        if 'constants' in data:
            model.update_constants(**data['constants'])
        if 'epsilon' in data:
            model.epsilon = data['epsilon']
        if 'density' in data:
            model.point_density = data['density']

        model.first_step(three_dimensional=False)

        return jsonify({
            'success': True,
            'points': model.points.tolist(),
            'normals': model.normals.tolist(),
            'step': 0,
            'config': {
                'epsilon': model.epsilon,
                'density': model.point_density
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/compute/step', methods=['POST'])
def compute_step():
    try:
        data = request.json

        model = nm.Model()

        x_func = data.get('x_func', 'x')
        y_func = data.get('y_func', 'y')
        model.update_function(x=x_func, y=y_func)

        if 'constants' in data:
            model.update_constants(**data['constants'])
        if 'epsilon' in data:
            model.epsilon = data['epsilon']
        if 'density' in data:
            model.point_density = data['density']

        model.points = np.array(data['points'], dtype=np.float64)
        model.normals = np.array(data['normals'], dtype=np.float64)
        model.prev_points = model.points.copy()
        model.prev_normals = model.normals.copy()

        current_step = data.get('current_step', 0)

        success = model.next_step()

        if not success:
            return jsonify({
                'success': True,
                'diverged': True,
                'step': current_step
            })

        if np.isnan(model.points).any() or np.isnan(model.normals).any():
            return jsonify({
                'success': True,
                'diverged': True,
                'step': current_step
            })

        return jsonify({
            'success': True,
            'diverged': False,
            'points': model.points.tolist(),
            'normals': model.normals.tolist(),
            'step': current_step + 1
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
    







