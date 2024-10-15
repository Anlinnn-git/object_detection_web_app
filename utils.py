import tensorflow as tf
from PIL import Image, ImageDraw, ImageColor, ImageFont, ImageOps
import numpy as np
import os

def resize_image(image_path, new_width=1024, new_height=1024):
    pil_image = Image.open(image_path)
    pil_image = ImageOps.fit(pil_image, (new_width, new_height), Image.LANCZOS)
    pil_image_rgb = pil_image.convert("RGB")
    filename = os.path.join('static/uploads', os.path.basename(image_path))  # Save resized image in static/uploads
    pil_image_rgb.save(filename, format="JPEG", quality=95)  # Use higher quality
    return filename

def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img

def draw_boxes(image, boxes, class_names, scores, max_boxes=20, min_score=0.3, save_path=None):
    colors = list(ImageColor.colormap.values())
    image_pil = Image.fromarray(np.uint8(image)).convert("RGB")

    try:
        font = ImageFont.truetype("arial.ttf", 25)
    except IOError:
        font = ImageFont.load_default()

    for i in range(min(boxes.shape[0], max_boxes)):
        if scores[i] >= min_score:
            ymin, xmin, ymax, xmax = tuple(boxes[i])
            display_str = "{}: {}%".format(class_names[i].decode("ascii"), int(100 * scores[i]))
            color = colors[hash(class_names[i]) % len(colors)]
            draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color, font, display_str_list=[display_str])

    # Save the image with boxes to the provided save_path
    if save_path is not None:
        image_pil.save(save_path, format="JPEG", quality=95)
    else:
        # If no save path is provided, save it to a default location
        save_path = save_image_with_boxes(image_pil)

    return save_path

def draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax, color, font, thickness=4, display_str_list=()):
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)], width=thickness, fill=color)

    display_str_heights = [font.getbbox(ds)[3] for ds in display_str_list]
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = top + total_display_str_height

    for display_str in display_str_list[::-1]:
        bbox = font.getbbox(display_str)
        text_width, text_height = bbox[2], bbox[3]
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin), (left + text_width, text_bottom)], fill=color)
        draw.text((left + margin, text_bottom - text_height - margin), display_str, fill="black", font=font)
        text_bottom -= text_height - 2 * margin

def save_image_with_boxes(image_pil):
    result_filename = os.path.join('static/uploads', 'result_image.jpg')  # Save in static/uploads
    image_pil.save(result_filename, format="JPEG", quality=95)
    return result_filename
