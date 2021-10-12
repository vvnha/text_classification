from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin

app = Flask(__name__)

# Apply Flask CORS
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/', methods=['GET'])
@cross_origin(origin='*')
def home():
    return 'HOME'

@app.route('/submit', methods=['POST'])
@cross_origin(origin='*')
def submit():
    # if 'file' not in request.files:
    #     flash('No file part')
    #     return redirect(request.url)
    file = request.files['file']

    form = request.form
    name = form['name']
    if (name is None):
        return 'Thieu name'
    return file.filename

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='6868')
