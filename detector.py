import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from utils import draw_boxes, load_img
import os
import time

# Load the TensorFlow Hub module (Faster RCNN)
module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
detector = hub.load(module_handle).signatures['default']

def run_detector(image_path):
    img = load_img(image_path)
    converted_img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]

    start_time = time.time()
    result = detector(converted_img)
    end_time = time.time()

    result = {key: value.numpy() for key, value in result.items()}

    print("Found %d objects." % len(result["detection_scores"]))
    print("Inference time: ", end_time - start_time)

    # Save the result image to static/uploads
    result_image_path = os.path.join('static/uploads', 'result_image.jpg')
    draw_boxes(img.numpy(), result["detection_boxes"], result["detection_class_entities"], result["detection_scores"], save_path=result_image_path)
    
    return 'result_image.jpg'

