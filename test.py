from PIL import Image
import os, scipy.misc

UPLOAD_FOLDER = 'static/images/uploads'


# open an image file (.bmp,.jpg,.png,.gif) you have in the working folder
CONTENT_FILENAME = '1.PNG'
imageFile = os.path.join(UPLOAD_FOLDER, CONTENT_FILENAME)
im1 = Image.open(imageFile)
# adjust width and height to your needs
width = 300
height = 225
# use one of these filter options to resize the image
im2 = im1.resize((width, height), Image.NEAREST)      # use nearest neighbour
im3 = im1.resize((width, height), Image.BILINEAR)     # linear interpolation in a 2x2 environment
im4 = im1.resize((width, height), Image.BICUBIC)      # cubic spline interpolation in a 4x4 environment
im5 = im1.resize((width, height), Image.ANTIALIAS)    # best down-sizing filter
ext = ".jpg"
print(scipy.misc.imread(imageFile).shape)
print(im1.size)
im2.save("NEAREST" + ext)
im3.save("BILINEAR" + ext)
im4.save("BICUBIC" + ext)
im5.save(os.path.join(UPLOAD_FOLDER, CONTENT_FILENAME))
