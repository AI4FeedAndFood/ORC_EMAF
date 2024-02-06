import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
from copy import deepcopy

OCR_HELPER_JSON_PATH  = r"CONFIG\\OCR_config.json"
CONFIG_DICT = json.load(open(OCR_HELPER_JSON_PATH, encoding="utf-8"))
 

def preprocessed_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    return thresh

class Template:
    """
    A class defining a template
    """
    def __init__(self, image_path, label, color, matching_threshold=0.5, transform_list=[lambda x:x]):
        """
        Args:
            image_path (str): path of the template image path
            label (str): the label corresponding to the template
            color (List[int]): the color associated with the label (to plot detections)
            matching_threshold (float): the minimum similarity score to consider an object is detected by template
                matching
        """
        self.image_path = image_path
        self.label = label
        self.color = color
        self.transform_list = transform_list
        self.template = preprocessed_image(cv2.imread(image_path))
        self.template_height, self.template_width = self.template.shape[:2]
        self.matching_threshold = matching_threshold
        
        self.transformed_template = self.transform(self.template, self.transform_list)
        
    def transform(cls, template, transform):
        return [trans(template) for trans in transform]

TRANSFORM = [lambda x: x, lambda x: cv2.resize(x, (int(x.shape[1]*1.05), x.shape[0])),
             lambda x: cv2.resize(x, (x.shape[1], int(x.shape[0]*1.05)))] # Maybe can be cleaner with a transform class

def visualize(cropped_image, filtered_objects):
    image_with_detections = deepcopy(cropped_image)
    for detection in filtered_objects:
        cv2.rectangle(
            image_with_detections,
            (detection["TOP_LEFT_X"], detection["TOP_LEFT_Y"]),
            (detection["BOTTOM_RIGHT_X"], detection["BOTTOM_RIGHT_Y"]),
            detection["COLOR"],2)
    plt.imshow(image_with_detections, cmap='gray')
    plt.show(block=True)

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
    return thresh

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

def non_max_suppression(objects, non_max_suppression_threshold=0.2, score_key="MATCH_VALUE"):
    """
    Filter objects overlapping with IoU over threshold by keeping only the one with maximum score.
    Args:
        objects (List[dict]): a list of objects dictionaries, with:
            {score_key} (float): the object score
            {top_left_x} (float): the top-left x-axis coordinate of the object bounding box
            {top_left_y} (float): the top-left y-axis coordinate of the object bounding box
            {bottom_right_x} (float): the bottom-right x-axis coordinate of the object bounding box
            {bottom_right_y} (float): the bottom-right y-axis coordinate of the object bounding box
        non_max_suppression_threshold (float): the minimum IoU value used to filter overlapping boxes when
            conducting non max suppression.
        score_key (str): score key in objects dicts
    Returns:
        List[dict]: the filtered list of dictionaries.
    """
    sorted_objects = sorted(objects, key=lambda obj: obj[score_key], reverse=True)
    filtered_objects = []
    for object_ in sorted_objects:
        overlap_found = False
        for filtered_object in filtered_objects:
            coord1 = [object_["TOP_LEFT_X"],object_["TOP_LEFT_Y"], object_["BOTTOM_RIGHT_X"], object_["BOTTOM_RIGHT_Y"]]
            coord2 = [filtered_object["TOP_LEFT_X"],filtered_object["TOP_LEFT_Y"],filtered_object ["BOTTOM_RIGHT_X"], filtered_object["BOTTOM_RIGHT_Y"]]
            iou = get_iou(coord1, coord2)
            if iou > non_max_suppression_threshold:
                overlap_found = True
                break
        if not overlap_found:
            filtered_objects.append(object_)
            
    return filtered_objects

def checkbox_match(templates, cropped_image):
    detections = []
    for i, template in enumerate(templates):
        w, h = template.template_width, template.template_height
        for transformed in template.transformed_template:
            template_matching = cv2.matchTemplate(transformed, cropped_image, cv2.TM_CCOEFF_NORMED)
            match_locations = np.where(template_matching >= template.matching_threshold)
        
            for (x, y) in zip(match_locations[1], match_locations[0]):
                match = {
                    "TOP_LEFT_X": x,
                    "TOP_LEFT_Y": y,
                    "BOTTOM_RIGHT_X": x + w,
                    "BOTTOM_RIGHT_Y": y + h,
                    "MATCH_VALUE": template_matching[y, x],
                    "LABEL" : template.label,
                    "COLOR": (0, 191, 255)
                }
                detections.append(match)

    return detections  

def get_checkboxes(cropped_image, templates, show=False):

    detections = checkbox_match(templates, cropped_image)
    filtered_detection = non_max_suppression(detections)
    if show : 
        visualize(cropped_image, filtered_detection)
    
    return filtered_detection

def detect_checkboxes(image, template_pathes, show=False):

    TRANSFORM = [lambda x: x]   
    
    templates = [Template(image_path=t_path, label="check", color=(0, 0, 0), matching_threshold=0.8, transform_list=TRANSFORM)
                 for t_path in template_pathes]        
    
    checkboxes = get_checkboxes(image, templates=templates, show=show)

    return checkboxes

def get_lines(bin_image, mode="vertical", show=False):
    (cst, var) = (0,1) if mode == "vertical"  else (1,0) # The specified axis (accoridig to np y,x) is the constant one
    image = bin_image.copy()
    ksize = (1,6) if mode == "vertical" else (6,1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=ksize)
    bin_image = cv2.dilate(bin_image, kernel, iterations=4)
    edges = cv2.Canny(bin_image,50,150,apertureSize=3)
 
    # Apply HoughLinesP method to
    # to directly obtain line end points
    lines_list =[]
    lines = cv2.HoughLinesP(
                edges, # Input edge image
                1, # Distance resolution in pixels
                np.pi/180, # Angle resolution in radians
                threshold=60, # Min number of votes for valid line
                minLineLength=200, # Min allowed length of line
                maxLineGap=300 # Max allowed gap between line for joining them ; Set according to the SEMAE format
                )
    
    for points in lines:
        # Extracted points nested in the list
        x1,y1,x2,y2=points[0]
        line  = [(x1,y1),(x2,y2)]
        if abs(line[0][cst]-line[1][cst])<40:
            lines_list.append(line)

    lines_list = extend_lines(lines_list, mode=mode)
   
    if show:
        img = bin_image.copy()
        value = 0 if show else 255
        for line in lines_list:
            [(x1,y1),(x2,y2)] = line
            cv2.line(img,(x1,y1),(x2,y2),value,6)
        
        plt.imshow(img, cmap="gray")
        plt.show()
    
    lines_list = sorted(lines_list, key=lambda l: l[0][cst])

    return lines_list 

def extend_lines(lines_list, mode="vertical"):
    """
    Extend Hough line along the "mode" axis
    """
    lines_list_c = lines_list.copy()
    (cst, var) = (0,1) if mode == "vertical"  else (1,0) # The specified axis (indicated with a 0) is the constant one (is just fomrmalism)
    clean_lines = []
    merged_line = []
    # Merge discontinious lines
    for i in range(len(lines_list_c)):
        if not i in merged_line :
            line_i = lines_list_c[i]
            for j in range(i+1, len(lines_list_c)):
                if not j in merged_line:
                    line_j = lines_list_c[j]
                    if abs((line_i[0][cst]+line_i[1][cst])/2 - (line_j[0][cst]+line_j[1][cst])/2)<10:
                        res_line = [[0,0], [0,0]]
                        merged_line.append(j)
                        axis_mean = int((line_i[0][cst]+line_i[1][cst])/2)
                        if line_i[0][var] < line_i[1][var]:
                            min_naxis = min(line_i[0][var] , line_j[0][var])
                            max_naxis = max(line_i[1][var] , line_j[1][var])
                            res_line[0][var], res_line[1][var] =  min_naxis, max_naxis
                        else: 
                            min_naxis = min(line_i[1][var] , line_j[1][var])
                            max_naxis = max(line_i[0][var] , line_j[0][var])
                            res_line[1][var], res_line[0][var] =  min_naxis, max_naxis
                        res_line[0][cst], res_line[1][cst] = axis_mean, axis_mean
                        lines_list_c[i] = res_line
                        line_i = lines_list_c[i]
                else: pass
            if abs(line_i[0][var] - line_i[1][var])>150:
                clean_lines.append(lines_list_c[i])
    return clean_lines

def delete_HoughLines(image, lines, show=False):
    img = image.copy()
    value = 0 if show else 255
    for line in lines:
        [(x1,y1),(x2,y2)] = line
        cv2.line(img,(x1,y1),(x2,y2),value,6)
    
    if show:
        plt.imshow(img)
        plt.show()
        
    return img

if __name__ == "__main__":
    from ProcessPDF import PDF_to_images, binarized_image
    print("start")
    path = r"C:\Users\CF6P\Desktop\ELPV\Data\scan4.pdf"
    images = PDF_to_images(path)
    images = images[0:]
    res_dict_per_image = {}
    for i, image in enumerate(images,1):
        print(f"\nImage {i} is starting")
        processed_image = binarized_image(image)

