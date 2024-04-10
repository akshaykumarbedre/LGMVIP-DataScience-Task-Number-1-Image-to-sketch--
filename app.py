from flask import Flask, request, render_template
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']

        if file:
            filename = secure_filename(file.filename)
            image_path = os.path.join('static\image_upload', filename)
            file.save(image_path)
            return convert_to_sketch(image_path)
    return render_template("index.html")

def convert_to_sketch(image_path):
    # Read the image
    img = cv2.imread(image_path)
    
    # Convert to grey scale
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Invert the image
    invert_img = cv2.bitwise_not(grey_img)
    
    # Blur the image
    blur_img = cv2.GaussianBlur(invert_img, (21, 21), sigmaX=0, sigmaY=0)
    
    # Invert the blurred image
    invblur_img = cv2.bitwise_not(blur_img)
    
    # Sketch
    sketch_img = cv2.divide(grey_img, invblur_img, scale=250.0)
    
    # Save the sketch
    sketch_path = os.path.join('static\saved_image', 'sketch.png')
    cv2.imwrite(sketch_path, sketch_img)
    
    # Return HTML to display the sketch
    return render_template("preview.html",sketch_path=sketch_path)

if __name__ == '__main__':
    app.run(debug=True)
