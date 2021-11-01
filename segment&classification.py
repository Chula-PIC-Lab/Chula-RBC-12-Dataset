import cv2
import numpy as np

from functools import cmp_to_key
import torch
from efficientnet_pytorch import EfficientNet
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms


sumAllBgPixel = [152, 159, 147]
NAME_OF_TYPES = ['00 Normal cell','01 Macrocyte','02 Microcyte','03 Spherocyte','04 Target cell','05 Stomatocyte','06 Ovalocyte','07 Teardrop','08 Burr cell','09 Schistocyte','10 uncategorised','11 Hypochromia']

# set variable
IMAGE_PATH = 'Dataset'
IMAGE_START_NUMBER = 1 # เริ่มรันจากลำดับรูปที่ x
IMAGE_END_NUMBER = 10 # รันจนถึงลำดับรูปที่ x

MODLE_PATH = 'rbc_efficientnetb1_12classes'



def custom_compare(x, y):
    ellipse = cv2.fitEllipseDirect(x)
    poly = cv2.ellipse2Poly((int(ellipse[0][0]), int(ellipse[0][1])), (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)), int(ellipse[2]), 0, 360, 5)
    ellipseAreax = cv2.contourArea(poly)
    
    ellipse = cv2.fitEllipseDirect(y)
    poly = cv2.ellipse2Poly((int(ellipse[0][0]), int(ellipse[0][1])), (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)), int(ellipse[2]), 0, 360, 5)
    ellipseAreay = cv2.contourArea(poly)
    return ellipseAreax > ellipseAreay

def compare_lenth(x,y):
    return len(x) > len(y)

def checkValidEllipse(ellipse):
    if ellipse[1][1]/ellipse[1][0] >= 2.0:
        return False
    return True 



# Load model
model_ft = EfficientNet.from_pretrained('efficientnet-b1', num_classes=12)
model_ft.load_state_dict(torch.load(MODLE_PATH,map_location='cpu'))
model_ft.eval()


for numImage in range(IMAGE_START_NUMBER,IMAGE_END_NUMBER+1):
    print(numImage)
    # Load image
    inputImage = cv2.imread(IMAGE_PATH+'/'+str(numImage)+'.jpg')
    inputImage = cv2.cvtColor(inputImage,cv2.COLOR_BGR2RGB)
    b,g,r = cv2.split(inputImage)
    greyImage = g.copy()

    # Preprocessing
    clahe = cv2.createCLAHE(clipLimit = 3.0, tileGridSize = (8,8))
    greyImage = clahe.apply(greyImage)
    blur = cv2.GaussianBlur(greyImage,(3,3),0)
    ret,th = cv2.threshold(blur,0,255,cv2.THRESH_OTSU)

    cv2.bitwise_not(th,th)
    x,y = th.shape
    th[0,:] = 0
    th[:,0] = 0
    th[x-1,:] = 0
    th[:,y-1] = 0

    threshold_image = th.copy()
    element_0 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    threshold_image = cv2.morphologyEx(threshold_image,cv2.MORPH_GRADIENT,element_0)

    # Find RBC contour
    contours,hierarchy  = cv2.findContours(threshold_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    newContour = list()
    height, width = inputImage.shape[:2]

    filled = np.zeros_like(b)
    cv2.fillPoly(filled, contours, color=(255,255,255))
    
    # Find normalize value
    mask = 255- filled
    mask = mask/255.0
    countAllPixel = cv2.countNonZero(mask)

    countFilled = cv2.sumElems(inputImage*np.reshape(mask,(480,640,1)))
    countFilled = np.array(countFilled)
    bg = countFilled/countAllPixel
    bg = bg[0:3]
    bg = np.ones((480,640,3))*bg
    bg = bg.astype(int)
    
    dif = sumAllBgPixel - bg
    
    
    concavePointImg = inputImage.copy()
    tempImage = np.zeros_like(g)
    index = 10
    startIndex = 3
    
    # In the loop below, Each contour is separated if it is an overlapping cells
    # contourOut variable is the output of this loop
    contourOut = []
    for c in contours:

        mask = np.zeros_like(g)
        
        cv2.drawContours(mask,[c],0,1,-1)
        if cv2.contourArea(c) < 500:
            continue
        if cv2.contourArea(c) < 1500:
            contourOut.append(c)
            continue
        
        
        
        concavepoints = np.zeros(len(c))
        for i in range(len(c)):   
            check = 1
            tempImage[c[i][0][1],c[i][0][0]] = 3
            tempDistance = 0
            for k in range(startIndex,index+1):
                x = (c[(i-k)%len(c)][0][0]+c[(i+k)%len(c)][0][0])//2
                y = (c[(i-k)%len(c)][0][1]+c[(i+k)%len(c)][0][1])//2
                dis = np.linalg.norm(c[i][0]-(x,y))
                if cv2.pointPolygonTest(c,(x,y),False) >= 0 or dis < tempDistance:
                    check = 0
                tempDistance = dis
            if check ==1:
                tempImage[c[i][0][1],c[i][0][0]] = 4
                concavepoints[i] = 1.0

        connect = 0
        listMax = [0,0,-1]
        interval = 15
        countGroup = 0


        newConcavePoints = np.zeros(len(c))
        startPoint = -1
        firstConcavePoint = -1
        checkRound = 0
        num = 0
        if 1.0 in concavepoints:
            while True:
                if concavepoints[num] == 1.0:
                    if firstConcavePoint == -1:
                        firstConcavePoint = num
                    elif firstConcavePoint == num:
                        checkRound = 1
                    if connect == 0:
                        startPoint = num
                        connect = 1
                else:
                    if connect == 1:
                        middlePoint = int((num+startPoint)/2)
                        connect = 0
                        tempImage[c[middlePoint][0][1],c[middlePoint][0][0]] = 4
                        cv2.circle(concavePointImg,(c[middlePoint][0][0],c[middlePoint][0][1]), 3, (255,0,0), -1)

                        newConcavePoints[middlePoint] = 1 
                    if firstConcavePoint < num and checkRound == 1:
                        break
                num = (num+1)%len(c)
        pointsForCheck = np.zeros(len(c))
        indexI = 0
        startPoint = -1
        ellipseContours = []
        
        if 1 in newConcavePoints[:]:
            while True:
                if newConcavePoints[indexI] == 1:
                    if startPoint == -1:
                        startPoint = indexI
                        indexI = (indexI+1)%len(c)
                        continue
                    if startPoint < indexI:
                        newContour = c[startPoint:indexI]
                        if len(newContour)>10:
                            ellipseContours.append(newContour)
                    else:
                        newContour = np.concatenate((c[startPoint:],c[:indexI]))
                        if len(newContour)>10:
                            ellipseContours.append(newContour)

                    if pointsForCheck[indexI] == 1:
                        break
                    startPoint = indexI
                    pointsForCheck[indexI] = 1
                indexI = (indexI+1)%len(c)

        ellipseContours = sorted(ellipseContours, key=lambda x: len(x),reverse=True)
        concavePointImg = inputImage.copy()
        for ellipseContour in ellipseContours:
            rcolor = np.random.randint(256)
            gcolor = np.random.randint(256)
            bcolor = np.random.randint(256)
            for point in ellipseContour:
                cv2.circle(concavePointImg,(point[0][0],point[0][1]), 1, (rcolor,gcolor,bcolor), -1)

        unvalidEllipseContours = []
        tempMask = mask.copy()
        ii = 0
        concavePointImg = inputImage.copy()
        for ellipseContour in ellipseContours:
            if cv2.contourArea(ellipseContour) < 30:
                continue
            ellipse = cv2.fitEllipseDirect(ellipseContour)
            poly = cv2.ellipse2Poly((int(ellipse[0][0]), int(ellipse[0][1])), (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)), int(ellipse[2]), 0, 360, 5)

            ellipseArea = cv2.contourArea(poly)
            
            ellipseMask = np.zeros_like(g)
            cv2.drawContours(ellipseMask,[poly],0,1,-1)

            if ellipseArea < 700:
                unvalidEllipseContours.append(ellipseContour)
                continue
            
            countMask = np.count_nonzero(np.logical_and(mask,ellipseMask))
            countTempMask = np.count_nonzero(np.logical_and(tempMask,ellipseMask))
            
            if float(countMask)/ellipseArea <0.8 or float(countTempMask)/ellipseArea <0.2:
                unvalidEllipseContours.append(ellipseContour)
                
                continue

                
            tempMask = np.logical_xor( tempMask ,np.logical_and(tempMask,ellipseMask))
            cv2.drawContours(concavePointImg,[poly],0,(0,255,0),1)
            contourOut.append(poly)
            ii = ii+1
        unvalidEllipseContours = sorted(unvalidEllipseContours, key=cmp_to_key(custom_compare))
        if len(unvalidEllipseContours) > 1:
            checkValid = [1]*len(unvalidEllipseContours)
            
            for i in range(len(unvalidEllipseContours)):
                for j in range(len(unvalidEllipseContours)):
                    if i != j and checkValid[i] == 1 and checkValid[j] == 1:
                        contour = np.concatenate((unvalidEllipseContours[i],unvalidEllipseContours[j]))
                        ellipse = cv2.fitEllipseDirect(contour)

                        if np.isnan(ellipse[1][0]) or np.isnan(ellipse[1][1]):
                            continue
                        poly = cv2.ellipse2Poly((int(ellipse[0][0]), int(ellipse[0][1])), (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)), int(ellipse[2]), 0, 360, 5)
                        ellipseArea = cv2.contourArea(poly)
                        
                        
                        ellipseMask = np.zeros_like(g)
                        cv2.drawContours(ellipseMask,[poly],0,1,-1)
                        countMask = np.count_nonzero(np.logical_and(mask,ellipseMask))
                        countTempMask = np.count_nonzero(np.logical_and(tempMask,ellipseMask))
                        

                        if ellipseArea < 500:
                            continue
                        if ellipse[1][1]/ellipse[1][0] >= 2.0:
                            continue
                        
                        if float(countMask)/ellipseArea <0.9 or float(countTempMask)/ellipseArea <0.2:
                            continue
                        if ellipseArea > 1700:
                            continue
                        else:
                            checkValid[i] = 0
                            checkValid[j] = 0
                            tempMask = np.logical_xor( tempMask ,np.logical_and(tempMask,ellipseMask))
                            cv2.drawContours(concavePointImg,[poly],0,(0,255,0),1)
                            contourOut.append(poly)
                            ii = ii+1


        contours,hierarchy  = cv2.findContours(tempMask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        if len(contours) == 1:

            contourOut.append(contours[0])
    
    # make single cell image for feeding to model
    ouptut = inputImage.copy()
    ouptut = ouptut+dif
    

    outputimage = inputImage.copy()

    # variables for writing results on output image
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (0, 185)
    fontScale = 1
    color = (0, 0, 255)
    thickness = 2
    numm = 0

    # loop all contours to predict each cell
    for c in contourOut:
        # print(contourOut)
        
        outputimage = cv2.drawContours(outputimage,[c],0,(0,255,0),1)

        imageTemp = np.zeros_like(threshold_image)
        
        cv2.fillPoly(imageTemp, [c], color=(255,255,255))
        imageTemp = cv2.bitwise_and(ouptut,ouptut,mask = imageTemp)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        # print(center)
        
        # centering the cell on image
        middleX = center[1]
        middleY = center[0]
        x0 = middleX-35
        x1 = middleX+37
        y0 = middleY-35
        y1 = middleY+37
        if x0 < 0 :
            x1 += abs(x0)
            x0 = 0
        elif x1>480:
            x0 -= x1%480
            x1 = 480
        if y0 < 0 :
            y1 += abs(y0)
            y0 = 0
        elif y1 >640 :
            y0 -= y1%640
            y1 = 640


        image = imageTemp[x0:x1,y0:y1]
        numm = numm+1
        
        # normalize and resize image
        image = image.astype(float)/255
        image = cv2.resize(image, (224,224))
        
        image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        image = torch.from_numpy(image).float().permute(2,0,1).unsqueeze(0)
        

        # predict and write on output image
        with torch.no_grad():
            predoutput = model_ft(image)
            _, preds = torch.max(predoutput, 1)
            outputimage = cv2.putText(outputimage, str(preds.numpy()[0]), center, font, fontScale, 
                    color, thickness, cv2.LINE_AA, False)

    # write output image file
    outputimage = cv2.cvtColor(outputimage,cv2.COLOR_RGB2BGR)
    cv2.imwrite('test'+str(numImage)+'.jpg',outputimage)



