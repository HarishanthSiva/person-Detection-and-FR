# USAGE
# python detect.py --images images

# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import matplotlib.pyplot as plt
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import numpy as np
import argparse
import imutils
import cv2
import dlib


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", help="path to images directory",
                default="/home/harishanth/GIT/pedestrian-detection/images")
ap.add_argument("-p", "--shape-predictor",
	help="path to facial landmark predictor",default= "/home/harishanth/GIT/pedestrian-detection/shape_predictor_68_face_landmarks.dat")
args = vars(ap.parse_args())

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# loop over the image paths
imagePaths = list(paths.list_images(args["images"]))

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
fa = FaceAligner(predictor, desiredFaceWidth=256)


for imagePath in imagePaths:
    # load the image and resize it to (1) reduce detection time
    # and (2) improve detection accuracy
    image = cv2.imread(imagePath)
    #image = imutils.resize(image, width=min(400, image.shape[1]))
    orig = image.copy()
    crop = image.copy()

    # detect people in the image
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
                                            padding=(8, 8), scale=1.05)

    # draw the original bounding boxes
    for (x, y, w, h) in rects:
        cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        x1 = xA
        x2 = xB
        y1 = yA
        y2 = yB

        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
        crop_img = crop[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

        plt.axis("off")
        plt.imshow(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
        plt.show()

        rects = detector(crop_img, 2)

        print(len(rects))
        for rect in rects:
        # extract the ROI of the *original* face, then align the face
        # using facial landmarks
            (x, y, w, h) = rect_to_bb(rect)
            faceOrig = imutils.resize(crop_img[y:y + h, x:x + w], width=256)
            faceAligned = fa.align(crop_img, gray, rect)

            import uuid

            f = str(uuid.uuid4())
            #cv2.imwrite("foo/" + f + ".png", faceAligned)

            plt.subplot(121)
            plt.axis("off")
            plt.imshow(cv2.cvtColor(faceOrig, cv2.COLOR_BGR2RGB))

            plt.title("Original")

            plt.subplot(122)
            plt.axis("off")
            plt.imshow(cv2.cvtColor(faceAligned, cv2.COLOR_BGR2RGB))
            # plt.imshow(faceAligned)
            plt.title("Aligned")
            plt.show()
    # cv2.waitKey(0)

    # show some information on the number of bounding boxes
    filename = imagePath[imagePath.rfind("/") + 1:]
    print("[INFO] {}: {} original boxes, {} after suppression".format(
        filename, len(rects), len(pick)))



    plt.axis("off")
    plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
    plt.show()
# cv2.imshow("After NMS", image)
# cv2.waitKey(0)
