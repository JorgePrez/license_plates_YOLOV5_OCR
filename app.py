from flask import Flask, render_template, request , json, jsonify
import os 
import json
import urllib.request
from werkzeug.utils import secure_filename

from deeplearning import object_detection
# webserver gateway interface
app = Flask(__name__)

BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH,'static/upload/')


#DEL MULTIPART
UPLOAD_FOLDER = 'static/upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

#ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



 # SITIO WEB
@app.route('/',methods=['POST','GET'])
def index():
    if request.method == 'POST':
        upload_file = request.files['image_name']
        filename = upload_file.filename
        path_save = os.path.join(UPLOAD_PATH,filename)
        upload_file.save(path_save)
        text_list, list_json = object_detection(path_save,filename)
        json_resultante= json.dumps(list_json)
        
        #print('TEXT LIST' + str(text_list))
        print(json_resultante)


        return render_template('index.html',upload=True,upload_image=filename,text=text_list,no=len(text_list))

    return render_template('index.html',upload=False)



# ENDPOINT, que se le envia foto y devuelve JSON con número de placa y bounding box con ubicación de placa
@app.route('/api/license_plate', methods=['POST'])
def upload_file():
    # check if the post request has the file part
    #print(request.files)
    if 'imagen_subida' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp
 
    files = request.files.getlist('imagen_subida')
     
    errors = {}
    success = False
     
    for file in files:      
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            success = True
            path_save = os.path.join(UPLOAD_PATH,filename)
            text_list, list_json = object_detection(path_save,filename)
            json_resultante= json.dumps(list_json)
            print(json_resultante)
        

        else:
            errors[file.filename] = 'File type is not allowed'
 
    if success and errors:
        errors['message'] = 'File(s) successfully uploaded'
        resp = jsonify(errors)
        resp.status_code = 500
        return resp
    if success:
        resp = jsonify(list_json)
        resp.status_code = 201
        return resp
    else:
        resp = jsonify(errors)
        resp.status_code = 500
        return resp



if __name__ =="__main__":
    app.run(debug=True)