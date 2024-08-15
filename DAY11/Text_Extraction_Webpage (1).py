import pytesseract
from flask import Flask, render_template, request, send_from_directory, jsonify
import os
import tempfile
import cv2
import numpy as np

app = Flask(__name__)

# Temporary directory for uploads

UPLOAD_FOLDER = tempfile.mkdtemp()

ALLOWED_EXTENSIONS = {'png','jpg','jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):

    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(input_path, output_path):

    # Read the image

    img = cv2.imread(input_path, cv2.IMREAD_COLOR)

    # Convert to grayscale

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce image noise and improve contour detection

    blurred = cv2.GaussianBlur(gray, (5,5), 0)

    # Adaptive thresholding

    thresh = cv2.adaptiveThreshold(blurred, 0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Dilation to enhance the main objects

    kernel = np.ones((3,3), np.uint8)

    dilated = cv2.dilate(thresh, kernel, iterations=0)

    # Convert to three channels to merge with original image

    colored_dilated = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)

    # Bitwise the dilated image with original to get highlighted text regions

    result = cv2.bitwise_or(img, colored_dilated)

    # Convert the result to grayscale

    final_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    # Invert the image, as the text is now white on a black background

    #final_result = cv2.bitwise_not(final_gray)

    # Save the processed image

    #cv2.imwrite(output_path, final_result)

    cv2.imwrite(output_path, final_gray)


    return output_path

@app.route('/')

def index():

    return render_template('index.html')

@app.route('/upload_files', methods=['POST'])

def upload_files():

    uploaded_files = request.files.getlist("files")

    filenames = []

    for file in uploaded_files:

        if file and allowed_file(file.filename):

            filename = os.path.join(app.config['UPLOAD_FOLDER'], os.path.basename(file.filename))

            file.save(filename)

            filenames.append(file.filename)

    return jsonify({"files": filenames})

@app.route('/get_image/<filename>', methods=['GET'])

def get_image(filename):

    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/get_image_info/<filename>', methods=['GET'])

def get_image_info(filename):

    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    processed_filename = "processed_" + filename

    output_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)

    # Process the image

    process_image(input_path, output_path)

    # Extract text from the processed image using Tesseract

    extracted_text = pytesseract.image_to_string(output_path, config="--psm 6")

    # Returning paths for both images and the extracted text

    return jsonify({"original": filename, "processed": processed_filename, "text": extracted_text})

    return jsonify({"original": filename, "processed": processed_filename})

if __name__ == '__main__':

    app.run(debug=True)