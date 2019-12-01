import cv2
import matplotlib.pyplot as plt
import numpy as np

def lastRow(gray_im):
    plt.imshow(gray_im, cmap="Greys_r")
    plt.show()

    inverted = 255*(gray_im < 128).astype(np.uint8) # To invert the text to white
    coords = cv2.findNonZero(inverted) # Find all non-zero points (text)
    x, y, w, h = cv2.boundingRect(coords) # Find minimum spanning bounding box
    rect = gray_im[y:y+h, x:x+w]
    cv2.imwrite("rect.png", rect)

    rowStepSize = 40
    bandHeight = 200
    lasti = 0

    subsums = []

    # shape of image is height by width
    gray_im = rect
    height = gray_im.shape[0]
    
    for i in range(height, int(height/2), -rowStepSize): #chang middle term back to 0
        if (i-rowStepSize >= 0):
            subimg = gray_im[i-bandHeight:i, :]
            subsum = np.sum(gray_im[i-bandHeight:i, :])
            subsums.append(subsum)

    lasti = 0
    for i in range(1, len(subsums)):
        if subsums[i] > subsums[i-1] and subsums[i] < subsums[i+1]:
            lasti = i
            break
    
    if lasti > 0:
    	wantedFirstRow = height - rowStepSize * lasti - bandHeight
    	if wantedFirstRow >= 0:
            wantedImg = gray_im[height - rowStepSize * lasti - bandHeight: height - rowStepSize * (lasti-1), :]
            plt.imshow(wantedImg, cmap="Greys_r")
            plt.show()
            return wantedImg
