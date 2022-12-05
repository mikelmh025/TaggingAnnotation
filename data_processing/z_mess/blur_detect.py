# import the necessary packages
# from imutils import paths
import argparse
import cv2
import data_utils
from pathlib import Path
import os


def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images",
	help="path to input directory of images",default="/Users/bytedance/Desktop/data/image datasets/fairface-img-margin125-trainval/fair_face_clean_debug")
ap.add_argument("-t", "--threshold", type=float, default=30.0,
	help="focus measures that fall below this value will be considered 'blurry'")
args = vars(ap.parse_args())

img_paths = data_utils.make_im_set(args['images'])


# loop over the input images
for imagePath in img_paths:
	# load the image, convert it to grayscale, and compute the
	# focus measure of the image using the Variance of Laplacian
	# method
	name = imagePath.split('/')[-1]
	

	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	fm = variance_of_laplacian(gray)
	text = "Not Blurry"
	# if the focus measure is less than the supplied threshold,
	# then the image should be considered "blurry"

	# save_name = name.split('.')[0]+'_'+str(int(fm))+'.jpg'
	save_name = str(int(fm))+'_'+name.split('.')[0]+'.jpg'
	if fm < args["threshold"]:
		save_dir = args['images']+'_blur'
	else:
		save_dir = args['images']+'_NOT_blur'
	Path(save_dir).mkdir(exist_ok=True, parents=True)
	save_path  = os.path.join(save_dir,save_name)
	cv2.imwrite(save_path, image)

	# # show the image
	# cv2.putText(image, "{}: {:.2f}".format(text, fm), (10, 30),
	# 	cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
	# cv2.imshow("Image", image)
	# key = cv2.waitKey(0)