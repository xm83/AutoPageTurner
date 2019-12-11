import cv2
import matplotlib.pyplot as plt
import numpy as np

def lastRow(gray_im):
    inverted = 255*(gray_im < 128).astype(np.uint8) # To invert the text to white
    coords = cv2.findNonZero(inverted) # Find all non-zero points (text)
    x, y, w, h = cv2.boundingRect(coords) # Find minimum spanning bounding box
    # if y + int(h/5) <= gray_im.shape[0]:
    rect = gray_im[y:y+h+20, :]
    cv2.imwrite("rect.png", rect)
    gray_im = rect

    rowStepSize = 40
    bandHeight = 200
    lasti = 0

    subsums = []

    # shape of image is height by width
    height = gray_im.shape[0]
    
    for i in range(height, int(height/2), -rowStepSize): #chang middle term back to 0
        if (i-rowStepSize >= 0):
            subimg = gray_im[i-bandHeight:i, :]
            subsum = np.sum(gray_im[i-bandHeight:i, :])
            print(subimg)
            subsums.append(subsum)
    print("subsums", subsums)
    lasti = 0
    for i in range(1, len(subsums)):
        if subsums[i] > subsums[i-1] and subsums[i] < subsums[i+1]:
            lasti = i
            break
    
    # --------------------------------------

    # for i in range(height, int(height/2), -rowStepSize): #chang middle term back to 0
    #     if (i-rowStepSize >= 0):
    #         subimg = gray_im[i-bandHeight:i, :]
    #         subsum = np.sum(subimg)
    #         subsums.append(subsum)
            
    # print("subsums", subsums)

    # for i in range(1, len(subsums)):
    #     if subsums[i] > subsums[i-1] and subsums[i] < subsums[i+1]:
    #         lasti = i
    #         break
    while lasti == 0:
        bandHeight-= 100
        rowStepSize -= 10

        [print(bandHeight, rowStepSize)]


        for i in range(height, int(height/2), -rowStepSize): #chang middle term back to 0
            if (i-rowStepSize >= 0):
                subimg = gray_im[i-bandHeight:i, :]
                subsum = np.sum(subimg)
                subsums.append(subsum)
                
        print("subsums", subsums)

        for i in range(1, len(subsums)):
            if subsums[i] > subsums[i-1] and subsums[i] < subsums[i+1]:
                lasti = i
                break


    print("lasti", lasti)
    if lasti > 0:
        wantedFirstRow = height - rowStepSize * lasti - bandHeight
        if wantedFirstRow >= 0:
            print("image height, width")
            print(gray_im.shape)
            print(height, gray_im.shape[1])
            print("top dimension, bottom dimension")
            print(height - rowStepSize * lasti - bandHeight)
            print(height - rowStepSize * (lasti-1))
            print(gray_im[height - rowStepSize * lasti - bandHeight: height - rowStepSize * (lasti-1), :])
            wantedImg = gray_im[height - rowStepSize * lasti - bandHeight: height - rowStepSize * (lasti-1), :]
            
            cv2.imwrite("lastRow.png", wantedImg)
            return wantedImg
        else:
            cv2.imwrite("lastRow.png", gray_im[height - rowStepSize*1])
            # cv2.imshow(gray_im[height - rowStepSize*1])

    return rect