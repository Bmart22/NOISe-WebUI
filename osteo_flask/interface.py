import os
import json
from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for, current_app, send_from_directory
)
from werkzeug.exceptions import abort
from werkzeug.utils import secure_filename

from . import unpatch_singular

#from flask import current_app

bp = Blueprint('interface', __name__)


@bp.route('/', methods=('GET', 'POST'))
def index():
    if request.method == 'POST':
        image = request.files['image']

        image.save(os.path.join(current_app.config['UPLOAD_FOLDER'], "submit.png"))

    
    
        return render_template('interface/index.html', image_exists=1)
        
    if request.method == 'GET':
        return render_template('interface/index.html', image_exists=0)
        
        
@bp.route('/results', methods=('GET', 'POST'))
def results():
    
    unpatch_singular.model_inference(img_foldername=current_app.config['UPLOAD_FOLDER'], out_foldername=os.path.join(current_app.config['UPLOAD_FOLDER'], "output"))
    
    return render_template('interface/results.html')

@bp.route('/src_image', methods=('GET', 'POST'))
def loadImage():
    print(current_app.config['UPLOAD_FOLDER'])
    return send_from_directory( current_app.config['UPLOAD_FOLDER'], "submit.png" )

@bp.route('/out_image', methods=('GET', 'POST'))
def loadOutputImage():
    print(current_app.config['UPLOAD_FOLDER'])
    return send_from_directory( os.path.join(current_app.config['UPLOAD_FOLDER'],"output"), "unpatch.png" )
