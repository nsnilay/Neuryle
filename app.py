from flask import Flask, render_template,  request, redirect, url_for, send_from_directory

import os
import sys
from PIL import Image

from werkzeug.utils import secure_filename

import scipy.io
import scipy.misc
from nst_utils import *
import numpy as np
import tensorflow as tf

UPLOAD_FOLDER = 'static/images/uploads' # folder where images are uploaded
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])
CONTENT_FILENAME = " "
STYLE_FILENAME = " "

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

def compute_content_cost(a_C, a_G):

    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape a_C and a_G (≈2 lines)
    a_C_unrolled = tf.reshape(tf.reshape((a_C),[-1]),[n_H*n_W,n_C])
    a_G_unrolled = tf.reshape(tf.reshape((a_G),[-1]),[n_H*n_W,n_C])

    # compute the cost with tensorflow (≈1 line)
    J_content = tf.reduce_sum(tf.square(tf.subtract(a_C,a_G)))/(4*n_H*n_W*n_C)

    return J_content

def gram_matrix(A):
    GA = tf.matmul(A,tf.transpose(A))
    return GA

def compute_layer_style_cost(a_S, a_G):


    # Retrieve dimensions from a_G (≈1 line)
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    # Reshape the images to have them of shape (n_C, n_H*n_W) (≈2 lines)
    a_S = tf.transpose(tf.reshape(tf.reshape((a_S),[-1]),[n_H*n_W,n_C]))
    a_G = tf.transpose(tf.reshape(tf.reshape((a_G),[-1]),[n_H*n_W,n_C]))
    # Computing gram_matrices for both images S and G (≈2 lines)
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    # Computing the loss (≈1 line)
    J_style_layer = tf.reduce_sum(tf.square(tf.subtract(GS,GG)))/(4*n_C*n_C*n_W*n_W*n_H*n_H)

    return J_style_layer

STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]

def compute_style_cost(sess, model, STYLE_LAYERS):

    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:

        # Select the output tensor of the currently selected layer
        out = model[layer_name]

        # Set a_S to be the hidden layer activation from the layer we have selected, by running the session on out
        a_S = sess.run(out)

        # Set a_G to be the hidden layer activation from same layer. Here, a_G references model[layer_name]
        # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
        # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
        a_G = out

        # Compute style_cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S, a_G)

        # Add coeff * J_style_layer of this layer to overall style cost
        J_style += coeff * J_style_layer

    return J_style

def total_cost(J_content, J_style, alpha = 10, beta = 40):

    ### START CODE HERE ### (≈1 line)
    J = alpha * J_content + beta * J_style
    ### END CODE HERE ###

    return J

def model_nn(sess, model, input_image, num_iterations = 140):

    # Initialize global variables (you need to run the session on the initializer)
    ### START CODE HERE ### (1 line)
    sess.run(tf.global_variables_initializer())
    ### END CODE HERE ###

    # Run the noisy input image (initial generated image) through the model. Use assign().
    ### START CODE HERE ### (1 line)
    sess.run(model['input'].assign(input_image))
    ### END CODE HERE ###

    for i in range(num_iterations):

        # Run the session on the train_step to minimize the total cost
        ### START CODE HERE ### (1 line)
        sess.run(train_step)
        ### END CODE HERE ###

        # Compute the generated image by running the session on the current model['input']
        ### START CODE HERE ### (1 line)
        generated_image = sess.run(model['input'])
        ### END CODE HERE ###

        # Print every 20 iteration.
        if i%20 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))

            # save current generated image in the "/output" directory
            save_image("output/" + str(i) + ".png", generated_image)

    # save last generated image
    save_image('output/generated_image.jpg', generated_image)

    return generated_image

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def reduce_dims(image_path):
    imageFile = image_path
    im1 = Image.open(imageFile)
    # adjust width and height to your needs
    width = 400
    height = 300
    im5 = im1.resize((width, height), Image.ANTIALIAS)    # best down-sizing filter
    ext = ".jpg"
    im5.save(image_path)

app.debug = True
@app.route('/', methods = ['POST','GET'])
def index():
    image_files = []
    content_full_filename = ' '
    style_full_filename = ' '
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file1' not in request.files:
            print('No file part')
            #return redirect(request.url)

        file = request.files['file']
        style_file = request.files['file1']

        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            print('No selected file')
        #    return redirect(request.url)
        if style_file.filename == '':
            print('No selected style file')
    #        return redirect(request.url)
        if file and allowed_file(file.filename):
            global CONTENT_FILENAME
            CONTENT_FILENAME = secure_filename(file.filename)
            print(CONTENT_FILENAME)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], CONTENT_FILENAME))
        #    return redirect(url_for('uploaded_file',
        #                            filename=filename))
        if style_file and allowed_file(style_file.filename):
            global STYLE_FILENAME
            STYLE_FILENAME = secure_filename(style_file.filename)
            print(STYLE_FILENAME)
            style_file.save(os.path.join(app.config['UPLOAD_FOLDER'], STYLE_FILENAME))

        content_full_filename = '../' + os.path.join(app.config['UPLOAD_FOLDER'], CONTENT_FILENAME)
        style_full_filename = '../' + os.path.join(app.config['UPLOAD_FOLDER'], STYLE_FILENAME)

    if request.method == 'GET':
        print("GET")
    return render_template ('index.html',  content_image = content_full_filename, style_image = style_full_filename) #This line will render files from the folder templates


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

#This line is for the link About that you will use to go to about page
@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/result',methods = ['POST', 'GET'])
def result():

    if(CONTENT_FILENAME==" "):
        return "Empty"

    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    reduce_dims(os.path.join(app.config['UPLOAD_FOLDER'], CONTENT_FILENAME))
    content_image = scipy.misc.imread(os.path.join(app.config['UPLOAD_FOLDER'], CONTENT_FILENAME))
    content_image = reshape_and_normalize_image(content_image)

    reduce_dims(os.path.join(app.config['UPLOAD_FOLDER'], STYLE_FILENAME))
    style_image = scipy.misc.imread(os.path.join(app.config['UPLOAD_FOLDER'], STYLE_FILENAME))
    style_image = reshape_and_normalize_image(style_image)

    generated_image = generate_noise_image(content_image)
    print("Images loaded...")

    model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")
    print("Model loaded...")

    sess.run(model['input'].assign(content_image))

    # Select the output tensor of layer conv4_2
    out = model['conv4_2']

    # Set a_C to be the hidden layer activation from the layer we have selected
    a_C = sess.run(out)

    # Set a_G to be the hidden layer activation from same layer. Here, a_G references model['conv4_2']
    # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
    # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
    a_G = out

    # Compute the content cost
    J_content = compute_content_cost(a_C, a_G)

    # Assign the input of the model to be the "style" image
    sess.run(model['input'].assign(style_image))

    # Compute the style cost
    J_style = compute_style_cost(sess, model, STYLE_LAYERS)

    J = total_cost(J_content,J_style, alpha=10, beta=40)

    # define optimizer (1 line)
    optimizer = tf.train.AdamOptimizer(2.0)

    # define train_step (1 line)
    train_step = optimizer.minimize(J)
    ("training..")

    sess.run(tf.global_variables_initializer())
    ### END CODE HERE ###

    # Run the noisy input image (initial generated image) through the model. Use assign().
    ### START CODE HERE ### (1 line)
    sess.run(model['input'].assign(generated_image))
    ### END CODE HERE ###

    for i in range(22):

        # Run the session on the train_step to minimize the total cost
        ### START CODE HERE ### (1 line)
        sess.run(train_step)
        ### END CODE HERE ###

        # Compute the generated image by running the session on the current model['input']
        ### START CODE HERE ### (1 line)
        generated_image = sess.run(model['input'])
        ### END CODE HERE ###

        # Print every 20 iteration.
        if i%20 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))

            # save current generated image in the "/output" directory
            save_image("static/images/output/" + str(i) + ".png", generated_image)

    # save last generated image
    save_image('static/images/output/generated_image.jpg', generated_image)


    if request.method == 'POST':
        result =  request.form
        return render_template('result.html', generated_image = '../' + "static/images/output/generated_image.jpg")


if __name__ == '__main__':
    app.run()
