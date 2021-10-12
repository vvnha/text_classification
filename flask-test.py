from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
import os
import testE
import test

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
    # -------
    file = request.files['file']

    # filename =  file.filename
    # form = request.form
    # name = form['name']
    # if (name is None):
    #     return 'Thieu name'

    # file.save(os.path.join('trash', filename))
    # f = open(os.path.join('trash', filename),'rb')
    # return file.read().decode("utf-16")
    # return file.read().decode('utf-8')

    content = file.read().decode('utf-8')
    # content = '''build a library system for saving materials
    # '''
    # content = '''	
    # Tài trợ thương mại Quốc tế và một số giải pháp để nâng cao hiệu quả hoạt động tài trợ thương mại Quốc tế của ngân hàng công thương Việt Nam'''

    rs = test(content)
    print(rs[0][1])
    return rs[0][1]

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='6868')
