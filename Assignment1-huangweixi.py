import cv2
import pytesseract
import numpy as np
from imutils.object_detection import non_max_suppression


# Configure the path to Tesseract
pytesseract.pytesseract.tesseract_cmd = r'D:\Tesseract\Tesseract-OCR\tesseract.exe'

# Turn on computer's front-facing camera
cap = cv2.VideoCapture(0)

#Enters a loop that continuously reads frames from the camera
while True:
    # Read a frame of video, ret is boolean, to indicate whether frame was successfully read
    ret, frame = cap.read()

    # Check if the frame was read successfully
    if not ret:
        break

    # load the input image and grab the image dimensions(H,W,C)
    image = frame
    (H, W) = image.shape[:2]
    print(H, W)

    # set the new width and height
    # calculate the ratio in change for resizing coordinates after detecting
    (newW, newH) = (512, 512)
    rate_W = W / float(newW)
    rate_H = H / float(newH)

    #process image in different way for comparing the effects of text detection and recognition
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # chose two output layer: probabilities and the Text box coordinates
    layernames = [ "feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

    # load the pre-trained EAST text detector
    net = cv2.dnn.readNet(r"C:\Users\50597\Desktop\frozen_east_text_detection.pb")

    # construct a blob for model and specify the parameters
    blob = cv2.dnn.blobFromImage(image, 1.0, (newW, newH),(123.68, 116.78, 103.94), swapRB=True, crop=False)

    #perform forward pass to obtain the two output layer sets
    net.setInput(blob)
    (scores, geometry) = net.forward(layernames)

    #print shape of output for follow-up process
    #since output shape of geometry is(1,5,128,128),we can know it run with RBOX（Rotated Bounding Box）
    print(scores.shape, geometry.shape)

    # grab the number of rows and columns from the scores volume
    # initialize the set of bounding box rectangles and corresponding confidence scores
    (numRows, numCols) = scores.shape[2:4]
    print(numRows, numCols)

    rects = []
    confidences = []

    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the geometrical data used to derive potential bounding box coordinates that surround text
        scoresData = scores[0, 0, y] # the possibility that the pixel on line y belong to text
        xData0 = geometry[0, 0, y] # The distance from the pixel on line y in the feature map to the border on the top of the text box
        xData1 = geometry[0, 1, y] # to the right border
        xData2 = geometry[0, 2, y] # to the bottom of the border
        xData3 = geometry[0, 3, y] # to the left border
        anglesData = geometry[0, 4, y] # the angle

        # loop over the number of columns
        for x in range(0, numCols):

            # if our score does not have sufficient probability, ignore it
            if scoresData[x] < 0.8:
                continue

            # compute the offset factor as our resulting feature maps will be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and compute the sin and cos
            angle = anglesData[x]
            print(angle)

            # check the angle
            cos = np.cos(angle)
            sin = np.sin(angle)
            print(cos, sin)

            # use the geometry volume to derive the width and height of the bounding box
            box_h = xData0[x] + xData2[x]
            box_w = xData1[x] + xData3[x]

            # compute both the starting and ending coordinates for the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) - (sin * xData2[x]))
            endY = int(offsetY + (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - box_w)
            startY = int(endY - box_h)

            # add the bounding box coordinates and probability score to respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        # ratios
        startX = max(0, int(startX * rate_W))
        startY = max(0, int(startY * rate_H))
        endX = min(int(endX * rate_W), W)
        endY = min(int(endY * rate_H), H)

        # draw the bounding box on the image
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        text_region = image[startY:endY, startX:endX]

        # use pytesseract for text recognition
        config = '--psm 6 --oem 3'
        text = pytesseract.image_to_string(text_region, lang='eng', config=config)
        cv2.putText(image, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # show the output image
    cv2.imshow("Text Detection and Recognition", image)

    # Press the 'Q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resource
cap.release()
cv2.destroyAllWindows()


