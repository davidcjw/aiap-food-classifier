import os
import numpy as np
import logging

# pylint: disable=E0401, W0611
from flask import Flask, render_template, url_for, request, jsonify
from flaskext.markdown import Markdown
from flask import send_from_directory
from waitress import serve
from inference import Inference

# Add logger steps
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')

file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.ERROR)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

# Instantiate our flask app
app = Flask(__name__)
port = int(os.environ.get("PORT", 5000))
Markdown(app)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# Instantiate model
inf = Inference()
inf.load_model()


@app.route('/', methods=['GET'])
def index():
    """Renders the index '/' endpoint of the webpage

    Returns:
        html template -- Rendered index html endpoint
    """
    return render_template('index.html')


@app.route('/info', methods=['GET'])
def short_description():
    """Renders the '/info' endpoint of the webpage

    Returns:
        {dictionary} -- JSON dictionary containing model information
    """
    return jsonify({
        'model': 'ResNet50',
        'input-size': '224x224x3',
        'num-classes': 12,
        'pretrained-on': 'ImageNet'
    })


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Returns the '/predict' endpoint

    Returns:
        {nested dictionary} -- returns a JSON dictionary containing labels and
        probability of each image (up to 3)
    """
    results = {}
    for i in range(len(request.files)):
        im = request.files['file'+str(i)]
        food_class, preds = inf.make_inference(im)
        label = food_class[np.argmax(preds[0])]
        proba = round(preds[0][np.argmax(preds[0])]*100, 1)
        results['food_'+str(i)] = {}
        results['food_'+str(i)]['food'] = label
        results['food_'+str(i)]['probability'] = str(proba) + '%'

    return jsonify(results)


@app.route('/docs', methods=['GET'])
def docs():
    """Returns the '/docs' endpoint

    Returns:
        html template -- Rendered docs html endpoint
    """
    return render_template('docs.html')


# This function is obsolete but kept for personal reference
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    target = os.path.join(APP_ROOT, 'static')
    if not os.path.isdir(target):
        os.mkdir(target)
    else:
        logger.error("Couldn't create upload directory: {}".format(target))
    for upload in request.files.getlist("image"):
        filename = upload.filename
        destination = "/".join([target, filename])
        upload.save(destination)

        food_class, preds = inf.make_inference(destination)
        label = food_class[np.argmax(preds[0])]
        proba = round(preds[0][np.argmax(preds[0])]*100, 0)
        logger.info("Uploaded {} to {}".format(filename, destination))

    return render_template('upload.html', image_name=filename,
                           pred=label, proba=proba)


# This function is obsolete but kept for personal reference
@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("static", filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=port)
    # For production mode, comment the line above and uncomment below
    # serve(app, host="0.0.0.0", port=8000)
