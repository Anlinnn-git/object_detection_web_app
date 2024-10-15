import os
from flask import Flask, render_template, request, redirect, url_for
from detector import run_detector
from utils import resize_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'  # Update this to point to the static/uploads folder

# Ensure static/uploads directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Resize image before processing
        resized_image_path = resize_image(filepath)

        # Run object detection on the image
        result_image_path = run_detector(resized_image_path)

        # Extract only the filename for display, not the full path
        return render_template('index.html', 
                               uploaded_image=file.filename, 
                               result_image=os.path.basename(result_image_path))

    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
