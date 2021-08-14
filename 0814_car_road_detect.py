# instruction
import os
import cv2
import numpy as np  
from matplotlib import pyplot as plt

def getLineCoordinatesFromParameters(image, line_parameters):
    slope = line_parameters[0]
    intercept = line_parameters[1]
    y1 = image.shape[0]       # since line will always start from bottom of image
    y2 = int(y1 * (3.4 / 5))  # some random point at 3/5
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    if x1<0 :
       x1 = 0
    print((x1,x2,y1,y2))
    return np.array([x1, y1, x2, y2])

def getSmoothLines(image, lines,threshold_slope):
    left_fit = []   # will hold m,c parameters for left side lines
    right_fit = []  # will hold m,c parameters for right side lines

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        #print(slope)
        if slope < -1*threshold_slope:
            left_fit.append((slope, intercept))
        elif slope > threshold_slope:
            right_fit.append((slope, intercept))
    
    if len(left_fit) == 0 or len(right_fit) == 0 :
       print("in")
       return 

    # axis = 0?
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    
    # now we have got m,c parameters for left and right line, we need to know x1,y1 x2,y2 parameters
    left_line = getLineCoordinatesFromParameters(image, left_fit_average)
    right_line = getLineCoordinatesFromParameters(image, right_fit_average)
    return np.array([left_line, right_line])

def canny_fun(image,sigma) :
    median = np.median(img_blur)
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))

    return cv2.Canny(image, lower, upper)

def mask_fun(image) :
    # mask
    # create a zero array 假如有些情況下，我們已經有一個矩陣，我們想生成一個跟它具有相同尺寸的零矩陣，我們可以用傳統的方法來建立，
    stencil = np.copy(image)*0 #creating a blank to draw lines on

    # specify coordinates of the polygon
    polygon = np.array([[0,1080], [500,650], [1420,650], [1700,1080]])

    # fill polygon with ones  用来填充凸多边形
    cv2.fillConvexPoly(stencil, polygon, (255,255,255))

    return cv2.bitwise_and(image, image, mask=stencil)
    #cv2.imshow('mask', mask)


def get_line(mask_image) :
    # HoughLinesP or not
    lines_canny_blur = cv2.HoughLinesP(mask_image, 0.3, np.pi/180, 60, np.array([]), minLineLength=50, maxLineGap=25) 
    return getSmoothLines(mask_image,lines_canny_blur,0.25);


def display_line(image,lines) :
    for line in lines :
        x1,y1,x2,y2 = line
        cv2.line(image,(x1,y1),(x2,y2),(255,0,0),3);
    return image


path = os.getcwd()
input_video_path = path + '\\影片偵測\\road_data_0709_Trim2.mp4'
cap = cv2.VideoCapture(input_video_path)
frame_list = []

while cap.isOpened():
    ret, frame = cap.read()
    
    if  ret == True :
        # read photo
        color_img = frame.copy()

        img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

        # blur
        img_blur = cv2.GaussianBlur(img,(5,5),0)

        # canny or not
        canny_blur = canny_fun(img_blur,0.95)

        # mask
        mask = mask_fun(canny_blur)

        mask_copy = mask.copy()

        lines = cv2.HoughLinesP(mask_copy, 0.3, np.pi/180, 60, np.array([]), minLineLength=50, maxLineGap=25) 
        for line in lines :
            x1,y1,x2,y2 = line[0]
            cv2.line(mask_copy,(x1,y1),(x2,y2),(255,0,0),3);
        cv2.imwrite(path+'\\step_photo\\mask_copy.png',mask_copy)
        #cv2.imshow('mask',mask)
        #plt.imshow(mask)
        #plt.show()

        try :
           smooth_line = get_line(mask)
           final_img = display_line(color_img,smooth_line)
           
           # add green area
           x1,y1,x2,y2 = smooth_line[0]
           x3,y3,x4,y4 = smooth_line[1]

           polygon = np.array([[x1,y1], [x2,y2], [x4,y4],[x3,y3]])

           cv2.fillConvexPoly(final_img, polygon, (0,230,0))

           frame_list.append(final_img)
           #cv2.imwrite(path+'\\step_photo\\final.png',final_img)
           #cv2.imshow('final_img',final_img)
        except :
           final_img = color_img.copy()
           frame_list.append(final_img)
           #cv2.imwrite(path+'\\step_photo\\final.png',color_img)
           #cv2.imshow('color_img',color_img)
    else :
           break
    
cap.release()

fps = 15
size = (1920,1080)

# write the video
video_path = path + '\\影片偵測\\'
out = cv2.VideoWriter(video_path + "car_road_detect_0814.avi",cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

for i in range(len(frame_list)):
    # writing to a image array
    out.write(frame_list[i])

out.release()
