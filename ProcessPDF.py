import os
import cv2
import fitz

import numpy as np
import matplotlib.pyplot as plt

from shutil import copyfile
from PIL import Image
import win32com

from imageTools import get_iou

def extract_from_mailbox(savePath, SenderEmailAddress, n_message_stop=50):

    outlook = win32com.client.Dispatch("Outlook.Application").GetNamespace("MAPI")
    inbox = outlook.GetDefaultFolder(6) 
    messages = inbox.Items
    messages.Sort("ReceivedTime", True)

    n_message = 0
    while n_message <= n_message_stop:
        message = messages[n_message]
        try:
            n_message+=1
            if message.Unread and message.SenderEmailAddress==SenderEmailAddress:
                # attachments = message.Attachments
                # attachment = attachments.Item(1)
                for attachment in message.Attachments:
                    attachment.SaveAsFile(os.path.join(savePath, str(attachment.FileName)))
                    if message.Unread:
                        message.Unread = False
                    break
        except:
            pass

def move_pdf(input_path, copy_path, rename = "", mode="same"):

    for pdf_path in get_all_pdfs_pathes(input_path):
        base, extension = os.path.splitext(os.path.split(pdf_path)[1])
        if rename:
            base=rename
        
        if mode == "same":
            new_name = base+extension

        copyfile(pdf_path, f"{copy_path}/{new_name}")
        os.remove(pdf_path)

def get_all_pdfs_pathes(dir_path):
    docs = os.listdir(dir_path)
    pdf_in_folder = [os.path.join(dir_path, file) for file in docs if os.path.splitext(file)[1].lower() == ".pdf"]
    return pdf_in_folder

def images_from_PDF(path, rot=False):
    """ 
    Open the pdf and return all pages as a list of array
    Args:
        path (path): python readable path
        POPPLER (path): Defaults to POPPLER_PATH.

    Returns:
        list of arrays: all pages as array
    """
    images = fitz.open(path)
    res = []
    for image in images:
        pix = image.get_pixmap(dpi=200)
        mode = "RGBA" if pix.alpha else "RGB"
        img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
        img = np.array(img)
        if rot:
            res.append(np.rot90(img, -1))
        else: 
            res.append(img)

    return res

def binarized_image(image):
    """ 
    Binarized one image thanks to OpenCV thersholding. niBlackThresholding has been tried.
    Args:
        image (np.array) : Input image

    Returns:
        np.array : binarized image
    """
    #image = image[3:-3, 3:-3]
    blur = cv2.bilateralFilter(image,5,200,200)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    adaptive_thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,33,10)
    return thresh

def adaptive_binarized_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    adaptive_thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,33,10)
    return adaptive_thresh

def get_iou(a, b, epsilon=1e-5):
    """ Given two boxes `a` and `b` defined as a list of four numbers:
            [x1,y1,x2,y2]
        where:
            x1,y1 represent the upper left corner
            x2,y2 represent the lower right corner
        It returns the Intersect of Union score for these two boxes.

    Args:
        a:          (list of 4 numbers) [x1,y1,x2,y2]
        b:          (list of 4 numbers) [x1,y1,x2,y2]
        epsilon:    (float) Small value to prevent division by zero

    Returns:
        (float) The Intersect of Union score.
    """
    # COORDINATES OF THE INTERSECTION BOX
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    # AREA OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)
    # handle case where there is NO overlap
    if (width<0) or (height <0):
        return 0.0
    area_overlap = width * height

    # COMBINED AREA
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined+epsilon)
    return iou

def get_main_rect(processed_image, rectangles):
    
    def _process_table_rectangles(rectangles):
        """
        Process rectangles for tables (table and landscape format)
        """  
        maxarea = 0
        for i, rect in enumerate(rectangles):
            if 45<rect[-1]:
                rectangles[i][1] = (rectangles[i][1][1], rectangles[i][1][0])
                if maxarea < rectangles[i][1][0]*rectangles[i][1][1]:
                    rot = 90-rect[-1]
            elif maxarea < rectangles[i][1][0]*rectangles[i][1][1]:
                rot = rect[-1] # The rot angle is chosen by taken the biggest rect angle
        
        xy_wh_rot = [[], [], []]
        for rect in rectangles:
            for comp in range(len(rect)):
                xy_wh_rot[comp].append(rect[comp])
              
        xmin_xmax_ymin_ymax = []
        for dist_i in [0,1]:
            for sens_j in [0,1]:
                if sens_j == 0 :
                    xmin_xmax_ymin_ymax.append(min([(rec[0][dist_i] - rec[1][dist_i]//2)+1 for rec in rectangles]))
                else :
                    xmin_xmax_ymin_ymax.append(max([(rec[0][dist_i] + rec[1][dist_i]//2)+1 for rec in rectangles]))
                    
        wh = (xmin_xmax_ymin_ymax[1]-xmin_xmax_ymin_ymax[0]+10, xmin_xmax_ymin_ymax[3]-xmin_xmax_ymin_ymax[2]+10)
        xy = (wh[0]//2 + xmin_xmax_ymin_ymax[0], wh[1]//2 + xmin_xmax_ymin_ymax[2])

        return (xy, wh, rot)
    
    y,x = processed_image.shape
    if len(rectangles)>2: # Clean if there is a black border of the scan wich is concider as a contour
        rectangles = [rect for rect in rectangles if not (0<x-rect[1][0]<10 or 0<y-rect[1][0]<10 or 0<x-rect[1][1]<0 or 0<y-rect[1][1]<10)]        
    rectangle = _process_table_rectangles(rectangles)

    return rectangle
    
def get_rectangles(processed_image, kernel_size=(3,3)):
    """
    Extract the minimum area rectangle containg the text. 
    Thanks to that detect if the image is a TABLE format or not.
    Args:
        processed_image (np.array): The binarized images
        kernel_size (tuple, optional): . Defaults to (3,3).
        interations (int, optional): _description_. Defaults to 2.

    Returns:
        format (str) : "table" or "other"
        rectangle (cv2.MinAreaRect) : The biggest rectangle of text found in the image
    """
    y, x = processed_image.shape[:2]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    dilate = cv2.dilate(~processed_image, kernel, iterations=2)
    contours,_ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print("No contour found : Crop is impossible, processed image is return.")
        return []

    rectangles = [cv2.minAreaRect(contour) for contour in contours]
    
    filtered_rects = []
    coord = lambda x: [int(x[0][0]-x[1][0]/2), int(x[0][1]-x[1][1]/2), int(x[0][0]+x[1][0]/2), int(x[0][1]+x[1][1]/2)]
    for rect in rectangles:
        rect=list(rect)
        if 45<rect[-1]: # Normalize to get x,y,w,h in the image refrential
                rect[1] = [rect[1][1], rect[1][0]]
                rect[-1] = rect[-1]-90
        overlap_found = False
        for f_rect in filtered_rects:
            coord1 = coord(rect)
            coord2 = coord(f_rect)
            iou = get_iou(coord1, coord2)
            if iou > 0.2 :
                overlap_found = True
                break
        if not overlap_found:
            filtered_rects.append(rect)
    
    rectangles =  [list(rect) for rect in rectangles if rect[1][0]>0.2*y] 
    
    return rectangles
    
def crop_and_adjust(processed_image, rect):
    """Crop the blank part around the found rectangle.

    Args:
        processed_image (np.array): The binarized image
        rect (cv2.MinAreaRect) : The biggest rectangle of text found in the image
    Returns:
        cropped_image (np.array) : The image cropped thanks to the rectangle
    """
    def _points_filter(points):
        """
        Get the endpoint along each axis
        """
        points[points < 0] = 0
        xpoints = sorted(points, key=lambda x:x[0])
        ypoints = sorted(points, key=lambda x:x[1])
        tpl_x0, tpl_x1 = xpoints[::len(xpoints)-1]
        tpl_y0, tpl_y1 = ypoints[::len(ypoints)-1]
        return tpl_y0[1], tpl_y1[1], tpl_x0[0], tpl_x1[0]
    
    if len(rect)==0 : 
        return processed_image

    box = np.intp(cv2.boxPoints(rect))    
    # Rotate image
    angle = rect[2]
    rows, cols = processed_image.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2,rows/2), angle, 1) # Rotation matrix
    img_rot = cv2.warpAffine(processed_image,M,(cols,rows))
    # plt.imshow(img_rot)
    # plt.show()
    # rotate bounding box, then crop
    rect_points = np.intp(cv2.transform(np.array([box]), M))[0] # points of the box after rotation
    y0, y1, x0, x1 = _points_filter(rect_points) # get corners
    cropped_image = img_rot[y0:y1, x0:x1]
    
    return cropped_image

def process_image(image, mode=""):

    bin_image = binarized_image(image)
    rects = get_rectangles(bin_image)
    main_rect = get_main_rect(bin_image, rects)

    if mode == "adaptive":
        adapt_image = adaptive_binarized_image(image)
        processed_image = crop_and_adjust(adapt_image, main_rect)

    else:
        processed_image = crop_and_adjust(bin_image, main_rect)

    return processed_image

if __name__ == "__main__":

    path = r"C:\Users\CF6P\Desktop\EMAF\DATA\scan11.pdf"
    images = images_from_PDF(path)
    for i, im in enumerate(images):
        im = process_image(im) if i%2==1 else im
        plt.imshow(im)
        plt.show()