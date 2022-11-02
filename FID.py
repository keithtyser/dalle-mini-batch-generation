import os
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy import array
from numpy.random import shuffle
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets.mnist import load_data
from skimage.transform import resize
from keras.datasets import cifar10
from keras.utils import load_img
from keras.utils import img_to_array
import csv


# scale an array of images to a new size


def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return asarray(images_list)

# calculate frechet inception distance


def calculate_fid(model, images1, images2):
    # calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = numpy.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


# prepare the inception v3 model
model = InceptionV3(include_top=False, pooling='avg',
                    input_shape=(299, 299, 3))
# # load cifar10 images
# (images1, _), (images2, _) = cifar10.load_data()
# shuffle(images1)
# images1 = images1[:10000


ground_paths = []

for filename in sorted(os.listdir("/Users/ktyser/Desktop/deep_learning/Ground Truth Images")):
    ground_paths.append(filename)

test_paths = []
for filename in os.listdir("/Users/ktyser/Desktop/deep_learning/Test Images"):
    test_paths.append(filename)

print(ground_paths)
print(test_paths)

FID_scores = []
for i in range(180):
    image1 = load_img(
        "/Users/ktyser/Desktop/deep_learning/Ground Truth Images/{}".format(ground_paths[i]))

    image1 = img_to_array(image1)
    image1 = array([image1, image1])

    image2 = load_img(
        "/Users/ktyser/Desktop/deep_learning/Test Images/{}".format(test_paths[i]))

    image2 = img_to_array(image2)
    image2 = array([image2, image2])
    print('Loaded', image1.shape, image2.shape)

    # convert integer to floating point values
    images1 = image1.astype('float32')
    images2 = image2.astype('float32')
    # resize images
    images1 = scale_images(images1, (299, 299, 3))
    images2 = scale_images(images2, (299, 299, 3))
    print('Scaled', images1.shape, images2.shape)
    # pre-process images
    images1 = preprocess_input(images1)
    images2 = preprocess_input(images2)
    # calculate fid
    fid = calculate_fid(model, images1, images2)
    print('FID: %.3f' % fid)
    FID_scores.append((ground_paths[i], test_paths[i], fid))

print(len(FID_scores))
print(FID_scores)

# note: If you use 'b' for the mode, you will get a TypeError
# under Python3. You can just use 'w' for Python 3

with open('/Users/ktyser/Desktop/deep_learning/fid_scores.csv', 'wb') as out:
    csv_out = csv.writer(out)
    csv_out.writerow(['ground', 'test', 'score'])
    for row in FID_scores:
        csv_out.writerow(row)
