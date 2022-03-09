# -*- coding:utf-8 -*-

import numpy as np
import os
from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
import matplotlib.pyplot as plt
import matplotlib
import io
from PIL import Image
from flask import Flask
import yaml
import random
import time
import base64

# append the path of detect_utils.py for obj detection
import sys
sys.path.append("../")
from detect_utils import ImgBox, draw_boxes


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])  # only image format allowed
CONFIG_LIST = os.listdir('config')  # config path
BOX_LABELS = ['x1', 'y1', 'x2', 'y2', 'Confidence', 'Class']  # used for table display
DPI = matplotlib.rcParams['figure.dpi']  # used for image original size display

# init app
app = Flask(__name__, template_folder='templates')
app.secret_key = "secret key"

# check uoload file format
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# load parameters from config file
def config_reader(config):

    # return params
    # weight: model; classes: class labels; img_size: reshaped size; color_bar: class-color match up
    with open(rf'{config}') as f:
        loader = yaml.load(f, Loader=yaml.FullLoader)
        class_num = loader['nc']
        weight = loader['weights']
        classes = loader['names']
        img_size = loader['img_size']
        colors = loader['colors']
    
    # generate colors randomly without repeatation if not set
    if not colors:
        colors = []
        for _ in range(class_num):
            while True:
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                if not color in colors:
                    colors.append(color)
                    break
    # pair colors with classes
    color_bar = dict(zip(classes, colors))

    return weight, classes, img_size, color_bar

# prevent favicon 404
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), \
                                'favicon.ico', mimetype='image/vnd.microsoft.icon')

# get drop list to select config
@app.route('/', methods=['GET'])
def drop_list():
    return render_template('idx.html', config_list=CONFIG_LIST)

# upload image, process and then display
@app.route('/', methods=['POST'])
def main_flow():

    # get uploaded file
    file = request.files['file']

    # locate config via drop list
    config = os.path.join('config', request.form.get('configs'))

    # no file uploaded
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)

    # image file uploaded
    if file and allowed_file(file.filename):

        # store image as stream and maintain original shape
        img = io.BytesIO()
        img_file = np.array(Image.open(file))
        img_h, img_w, img_d = img_file.shape
        fig_size = img_w / float(DPI), img_h / float(DPI)
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        ax.imshow(img_file)
        plt.savefig(img, format='png')
        img.seek(0)

        # trans image to base64
        img_url = base64.b64encode(img.getvalue()).decode()
        flash(f'Image displayed. Original size: {img_w}x{img_h}x{img_d}')

        # object detection and count time consuming
        try:
            start_time = time.time()
            img_copy = np.array(img_file)

            # load params
            weight, classes, img_size, color_bar = config_reader(config)

            # init class
            target_box = ImgBox(weight=weight, img_sz=img_size, names=classes, image=img_copy)

            # object detection and return resized image and objects info
            img_resized, box_info = target_box.get_img_boxed()

            # draw bounding boxes into resized image
            painted_img = draw_boxes(img_resized, box_info, color_bar)

            # store processed image as stream and maintain resized shape
            processed_img = io.BytesIO()
            processed_h, processed_w, processed_d = painted_img.shape
            processed_fig_size = processed_w / float(DPI), processed_h / float(DPI)
            processed_fig = plt.figure(figsize=processed_fig_size)
            processed_ax = processed_fig.add_axes([0, 0, 1, 1])
            processed_ax.axis('off')
            processed_ax.imshow(painted_img)
            plt.savefig(processed_img, format='png')
            processed_img.seek(0)

            # trans processed image to base64
            processed_img_url = base64.b64encode(processed_img.getvalue()).decode()

            # add labels for table display
            box_info.insert(0, BOX_LABELS)

        except Exception as e:
            flash('System error')
            app.logger.error('%s', e)

        else:
            end_time = time.time()
            time_cost = f'{end_time - start_time:.3f}s'
            flash(f'Detection succeeded! {time_cost} used! Processed image size: {processed_w}x{processed_h}x{processed_d}')
            app.logger.info('Image processed successfully!')
            return render_template('idx.html', img_url=img_url, \
                                        processed_img_url=processed_img_url, \
                                        config_list=CONFIG_LIST, \
                                        box_info=box_info)
        
        return render_template('idx.html', img_url=img_url, config_list=CONFIG_LIST)


