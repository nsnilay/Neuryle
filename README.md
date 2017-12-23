# Neuryle

This is a python flask based webapp on Neural Style Transfer inspired by deeplearning.ai's Course 4 week 4 assignment.

To run it locally, first clone the directory - 
``` git clone https://github.com/nsnilay/Neuryle.git ```

go into the directory - 
``` cd Neuryle ```

Install the dependencies
``` sudo pip install -r requirements.txt ```

Download imagenet-vgg-verydeep-19 weights from [here](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat) and save the file in ``` pretrained-model/ ``` directory

Then run
``` python app.py ```

Go to ``` localhost:5000 ``` in your web browser

Select the User image i.e. the input image and then the Style image (the image whose style you need to put on the user image)
Upload the images. (Currently, it supports jpg images)
then click the 'Stylize' button
