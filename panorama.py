import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

'''pathes for images'''
leftImagePath = sys.argv[1]
rightImagePath = sys.argv[2]
outputImagePath = sys.argv[3]

inputLeftImage = os.path.split(leftImagePath)
inputRightImage = os.path.split(rightImagePath)
outputImage = inputLeftImage[-1].replace(".jpg", "+").replace(
    ".png", "+")+inputRightImage[-1].replace(".", "_panoramic.")

# leftImagePath = './5/left.jpg'
# rightImagePath = './5/right.jpg'

'''actual images read'''
leftImage = cv2.imread(leftImagePath)
rightImage = cv2.imread(rightImagePath)
if leftImagePath is None or rightImage is None:
    print("Error reading the file")
    exit(1)

'''finding the scale for the height syncronization'''
minHeight = min(leftImage.shape[0], rightImage.shape[0])
maxHeight = max(leftImage.shape[0], rightImage.shape[0])
scale = maxHeight/minHeight

'''change the heigher image and save its H:W scale'''
if minHeight == rightImage.shape[0]:
    dsize = (
        round(leftImage.shape[1]/scale),
        round(leftImage.shape[0]/scale),
    )
    leftImage = cv2.resize(leftImage, dsize, interpolation=cv2.INTER_AREA)
else:
    dsize = (
        round(rightImage.shape[1]/scale),
        round(rightImage.shape[0]/scale),
    )
    rightImage = cv2.resize(rightImage, dsize, interpolation=cv2.INTER_AREA)


'''both images as gray'''
leftImage_gray = cv2.cvtColor(leftImage, cv2.COLOR_BGR2GRAY)
rightImage_gray = cv2.cvtColor(rightImage, cv2.COLOR_BGR2GRAY)

''' finding key point and dicriptors for both images according orb method'''
orb = cv2.ORB_create()
leftKeypoint, leftDiscriptors = orb.detectAndCompute(leftImage_gray, None)
rightKeypoint, rightDiscriptors = orb.detectAndCompute(rightImage_gray, None)

'''finding the 20% double checked minimum distance's matcher'''
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = matcher.match(leftDiscriptors, rightDiscriptors)
reducedMatches = sorted(matches, key=lambda x: x.distance)[
    :len(matches)//5]

'''for checking the matches'''
# imMatches = cv2.drawMatches(leftImage, leftKeypoint,
#                             rightImage, rightKeypoint, reducedMatches, None)

'''finding points for the homography'''
sourcePoints = np.float32(
    [leftKeypoint[match.queryIdx].pt for match in reducedMatches]).reshape(-1, 1, 2)
destinationPoints = np.float32(
    [rightKeypoint[match.trainIdx].pt for match in reducedMatches]).reshape(-1, 1, 2)
homography, _ = cv2.findHomography(destinationPoints, sourcePoints, cv2.RANSAC)

'''warp according to the homography'''
result = cv2.warpPerspective(rightImage, homography, ((
    rightImage.shape[1]+leftImage.shape[1]), leftImage.shape[0]))
result[0:leftImage.shape[0], 0:leftImage.shape[1]] = leftImage

'''remove the black area by finding contours of the rest of the image support angles lines'''
# make white image at the same size
whiteLeft = cv2.bitwise_not(cv2.bitwise_xor(leftImage, leftImage))
whiteRight = cv2.bitwise_not(cv2.bitwise_xor(rightImage, rightImage))

# make white panoramic with black background
white_result = cv2.warpPerspective(whiteRight, homography, ((
    whiteRight.shape[1]+whiteLeft.shape[1]), whiteLeft.shape[0]))
white_result[0:whiteLeft.shape[0], 0:whiteLeft.shape[1]] = whiteLeft

# make black panoramic with white background
white_result = cv2.bitwise_not(white_result)

# finding contours of white background
result_gray = cv2.cvtColor(white_result, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(result_gray, 1, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(
    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# crop the result according to contour location
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)

    '''for checking the rectangle'''
    # cv2.rectangle(white_result, (x, y), (x+w, y+h), (255, 0, 0), 30)

    # top left
    x1, y1, = x, y
    # bottom left
    x2, y2, = x, y+h
    # bottom right
    x3, y3, = x+w, y+h
    # top right
    x4, y4 = x+w, y

    # crop left
    if x3 == x4 == white_result.shape[1]:
        white_result = white_result[0:y2, 0:x1]
        result = result[0:y2, 0:x1]
    # crop bottom
    elif y2 == y3 == white_result.shape[0]:
        white_result = white_result[0:y1, 0:white_result.shape[1]]
        result = result[0:y1, 0:result.shape[1]]
    # crop top
    elif y1 == y4 == 0:
        white_result = white_result[y2:white_result.shape[0],
                                    0:white_result.shape[1]]
        result = result[y2:result.shape[0], 0:result.shape[1]]


# plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
# plt.show()

cv2.imwrite(os.path.join(outputImagePath, outputImage), result)
