import time

import json
import cv2
import openpyxl
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
import os
from unidecode import unidecode

import locale
locale.setlocale(locale.LC_TIME,'fr_FR.UTF-8')
from datetime import datetime
year = datetime.now().year

from paddleocr import PaddleOCR

from JaroDistance import jaro_distance
from imageTools import get_checkboxes, get_lines, delete_HoughLines
from ProcessPDF import images_from_PDF, process_image, get_all_pdfs_pathes, move_pdf

# CHNAGE THIS PATH TO THE CURRENT LOCATION 
application_path = os.path.realpath(os.path.dirname(__file__))

CONFIG_JSON_PATH  = os.path.join(application_path, "CONFIG\OCR_config.json")
CONFIG = json.load(open(CONFIG_JSON_PATH, encoding="utf-8"))

CHECKBOXES = os.path.join(application_path, CONFIG["checkboxes"])
LISTS = os.path.join(application_path, CONFIG["lists"])

class Row:
    def __init__(self, y_top, y_bot, row_text, checkboxes):
        self.row_text = row_text
        self.y_top = y_top
        self.y_bot = y_bot
        self.checkboxes = checkboxes
        self.header = None
        self.duplicate = False

class Column:
    """
    A class defining a template
    """
    def __init__(self, name, x_left, x_right):
        """
        Args:
            name (str): path of the template image path
            absolute_position (str): the label corresponding to the template
            theorical_col (List[float]): ratio of the horiontal position of the col compared with the image shape
        """
        self.name = name
        self.x_left = x_left
        self.x_right = x_right

def row_by_text(raw_texts, eps=35):
    """
    Sorts points following a "reading" rule. From top to bottom, from left to right. 

    Args:
        points (list of points : [(x,y)]): (x,y) for (horizontal, vertical)
        eps (int, optional): The threshold between two "lines" . Defaults to 200.
    Returns:
        sorted_indexes (list of int): Ordered point's indices
    """
    def _distance_line_point(A, B, point):
        (xA,yA), (xB,yB), (x,y) = A, B, point
        return abs((xB-xA)*(yA-y)-(xA-x)*(yB-yA))/((xB-xA)**2 + (yB-yA)**2+0.001)**0.5
    
    points_to_search = [[index,[(text["box"][0]+text["box"][2])*0.5, (text["box"][1]+text["box"][3])*0.5]] for index,text in enumerate(raw_texts)] # Tuple (index, coordinate)
    row_lines = []
    while len(points_to_search)>0:
        top_left, top_right = min(points_to_search, key=lambda p : p[1][0]+p[1][1])[1], max(points_to_search, key=lambda p : p[1][0]-p[1][1])[1] # Find the upperline
        
        if abs(top_left[1] - top_right[1])>=eps:
            M = min(top_left[1], top_right[1], top_left[1], top_right[1])
            top_left[1], top_right[1] = M, M

        new_line = []
        for index_point in points_to_search:
            if _distance_line_point(top_left, top_right, index_point[1])<eps: # Assign close enough points
                new_line.append(index_point)
        new_line = sorted(new_line, key=lambda i_p: i_p[1][0]) # Sort from left to right in the same line
        texts = [raw_texts[i_p[0]] for i_p in new_line]
        row_lines.append({
            "text" : texts,
            "y_top" : min([txt["box"][1] for txt in texts])-eps/2,
            "y_bot" : max([txt["box"][3] for txt in texts])+eps/2
            })
        points_to_search = [point for point in points_to_search if not point in new_line]

    return row_lines

def get_structural_elements(order_image, raw_image, y_im, eps=50, show=False):
    
    lines = {
        "vertical" : get_lines(order_image, mode="vertical", show=show),
        "horizontal" : get_lines(order_image, mode="horizontal", show=show)
        }
    
    min_dist = min([abs(l[0][1]-y_im/2) for l in lines["horizontal"]]) # Max distance between two horizontal lines

    if min_dist>100:
        adapt_order_image = process_image(raw_image, mode="adaptive")
        lines = {
        "vertical" : get_lines(adapt_order_image, mode="vertical", show=show),
        "horizontal" : get_lines(adapt_order_image, mode="horizontal", show=show)
        }
    
    # Extract horizontal lines and checkboxes
    template_pathes = [os.path.join(CHECKBOXES, dir) for dir in os.listdir(CHECKBOXES) if os.path.splitext(dir)[1].lower() in [".png", ".jpg"]]
    checkboxes = get_checkboxes(order_image, template_pathes=template_pathes, show=False) # List of checkbox dict {"TOP_LEFT_X"...}
        
    # Delete lines that come from checkboxes
    cleaned_lines = [lines["vertical"][0]]
    for i in range(1, len(lines["vertical"])):
        if abs(lines["vertical"][i][0][0]- lines["vertical"][i-1][0][0])>eps:
            cleaned_lines.append(lines["vertical"][i])
    lines["vertical"] = cleaned_lines

    return checkboxes, lines

def clean_image(image, lines, checkboxes):
    lineles_image = delete_HoughLines(image, lines)
    for check in checkboxes:
        image[check["TOP_LEFT_Y"]:check["BOTTOM_RIGHT_Y"], check["TOP_LEFT_X"]:check["BOTTOM_RIGHT_X"]] = 1
    return image

def paddle_OCR(image, show=False):
    def _cleanPaddleOCR(OCR_text):
        res = []
        for line in OCR_text:
            for t in line:
                    model_dict = {
                        "text" : "",
                        "box" : [],
                        "proba" : 0
                    }
                    model_dict["text"] = t[1][0]        
                    model_dict["box"] = t[0][0]+t[0][2]
                    model_dict["proba"] = t[1][1]
                    res.append(model_dict)
        return res

    ocr = PaddleOCR(use_angle_cls=True, lang='fr', show_log = False) # need to run only once to download and load model into memory
    results = ocr.ocr(image, cls=True)
    results = _cleanPaddleOCR(results)

    if show:
        im = deepcopy(image)
        for i, cell in enumerate(results):
            x1,y1,x2,y2 = cell["box"]
            cv2.rectangle(
                im,
                (int(x1),int(y1)),
                (int(x2),int(y2)),
                (0,0,0),2)
        plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        plt.show()

    return results

def _clean_text_rows(rows, x_im):

    for i, row in enumerate(rows):
        cleaned_row_text = []
        row_text = row.row_text
        for _, text in enumerate(row_text):
            if jaro_distance(unidecode(text["text"]), 'nb vol. a nombre verifier pesees')>0.85:
                mid_x = (text["box"][0]+text["box"][2])*0.5
                t_1 = deepcopy(text)
                t_1["text"] = "Nb vol. à verifier"
                t_1["box"][2] = mid_x
                cleaned_row_text.append(t_1)

                t_2 = deepcopy(text)
                t_2["text"] = "Nombre pesées"
                t_2["box"][0] = mid_x
                cleaned_row_text.append(t_2)
            
            elif jaro_distance(unidecode(text["text"]), 'n° de serie identification')>0.85:
                mid_x = (text["box"][0]+text["box"][2])*0.5
                t_1 = deepcopy(text)
                t_1["text"] = "Nb° de série"
                t_1["box"][2] = mid_x
                cleaned_row_text.append(t_1)

                t_2 = deepcopy(text)
                t_2["text"] = "identification"
                t_2["box"][0] = mid_x
                cleaned_row_text.append(t_2)

            elif len(text["text"])==1 and text["box"][2]>x_im*0.5:
                pass

            else:
                cleaned_row_text.append(text)

        rows[i].row_text = cleaned_row_text

    return rows

def get_text_lines(lines, checkboxes, raw_texts, x_im, y_im, delta=20):
    def _process_row_checkboxes(row_checkboxes, delta=20):
        col_checks = []
        while len(row_checkboxes)>0:
            first_check = min(row_checkboxes, key=lambda c: c["TOP_LEFT_X"])
            new_check = []
            for check in row_checkboxes:
                if abs(first_check["TOP_LEFT_X"]-check["TOP_LEFT_X"])<delta: # Assign close enough points
                    new_check.append(check)
            row_checkboxes = [check for check in row_checkboxes if not check in new_check]
            col_checks.append(new_check)
        return col_checks

    # Set rows
    h_lines = lines["horizontal"]
    min_dist = min([abs(l[0][1]-y_im/2) for l in lines["horizontal"]]) # min distance between a line and the middle of the image
    if min_dist<100:
        # print("Rows set with lines")
        row_texts = []
        paired_rows = []
        for i in range(len(h_lines)-1):
            upper_line, bottom_line = h_lines[i], h_lines[i+1]
            y_top, y_bot = max(upper_line[0][1], upper_line[1][1]), min(bottom_line[0][1], bottom_line[1][1])
            if abs(y_top-y_bot)>delta:
                # select texts between  lines
                row_text = [text for text in raw_texts if y_top<(text["box"][1]+text["box"][3])*0.5<y_bot and not text in paired_rows]
                row_texts.append({
                    "text" : row_text,
                    "y_top" : y_top,
                    "y_bot" : y_bot
                })
    else :
        # print("Rows set with text")
        row_texts =  row_by_text(raw_texts, eps=30)

    row_lines = []
    paired_rows = []
    for row_text in row_texts:
        if len(row_text["text"])>10:
            # Aglomerate text
            texts, y_top, y_bot = list(row_text.values())
            texts = sorted(texts, key=lambda t: t["box"][1])
            aglo_row = []
            paired_rows+=texts
            while len(texts)>0:
                ref_text = texts[0]
                aglo_text = []
                for text in texts:
                    if (ref_text["box"][0]<(text["box"][0]+text["box"][2])*0.5<ref_text["box"][2]) or (text["box"][0]<(ref_text["box"][0]+ref_text["box"][2])*0.5<text["box"][2]):
                        aglo_text.append(text)
                
                texts = [t for t in texts if not t in aglo_text]
                new_text = {
                "text" : unidecode(" ".join([str(t["text"]) for t in aglo_text])),
                "box" : [min([t["box"][0] for t in aglo_text]), min([t["box"][1] for t in aglo_text]), max([t["box"][2] for t in aglo_text]), max([t["box"][3] for t in aglo_text])],
                "proba" : min([t["proba"] for t in aglo_text])
                }
                aglo_row.append(new_text) 


            # Check if there are ticked checkboxes
            framed_checkboxes = [check for check in checkboxes if y_top<(check["TOP_LEFT_Y"]+check["BOTTOM_RIGHT_Y"])/2<y_bot]
            framed_checkboxes = _process_row_checkboxes(framed_checkboxes, delta=20)
            
            aglo_row = sorted(aglo_row, key=lambda t: t["box"][0])
            line = Row(y_top, y_bot, aglo_row, framed_checkboxes)
            row_lines.append(line)

    row_lines = _clean_text_rows(row_lines, x_im)

    # Copy lines with two "Type d'identification"
    copied_lines = []
    for i, row in enumerate(row_lines):
        
        if jaro_distance(unidecode(row.row_text[0]["text"]).lower(), "fabricant")>0.85 or jaro_distance(unidecode(row.row_text[1]["text"]).lower(), "modele")>0.85:
            row.header=True
            copied_lines.append(row)

        elif row.checkboxes:
            if len(row.checkboxes[0])>1:
                check_copy = sorted(row.checkboxes[0], key = lambda c: c["TOP_LEFT_Y"])
                row_c = deepcopy(row)
                row_c.checkboxes[0] = [check_copy[1]] # Second check
                row_c.duplicate = True
                row.checkboxes[0] = [check_copy[0]] # First check

                copied_lines.append(row)
                copied_lines.append(row_c)

            else:
                copied_lines.append(row)


    return copied_lines

def get_columns(lines, rows):
    def _line_shortcut(x_l, x_r):
        for r, row in enumerate(rows):
            for text in row.row_text:
                if (x_l<(text["box"][0]+text["box"][2])*0.5<x_r):
                    return (x_l,x_r)

    col_names = list(CONFIG["columns"].keys())
    columns = []

    # Else use lines that contains text 
    v_lines = lines["vertical"]
    matched_l_r = []
    for i in range(len(v_lines)-1):
        left_line, right_line = v_lines[i], v_lines[i+1]
        x_l, x_r = max(left_line[0][0], left_line[1][0]), min(right_line[0][0], right_line[1][0])
        res = _line_shortcut(x_l, x_r)
        if res:
            matched_l_r.append(res)

    if len(matched_l_r)==len(col_names):
        for i in range(len(col_names)):
            columns.append(Column(col_names[i], matched_l_r[i][0], matched_l_r[i][1]))
        # print("Columns are set with lines")
        return columns
        
    # Set the set of columns thanks to rows
    for row in rows:
        if len(row.row_text)==len(col_names):
            for i in range(len(col_names)):
                columns.append(Column(col_names[i], row.row_text[i]["box"][0], row.row_text[i]["box"][2]))
            # print("Columns are set with text")
            return columns
        
    else:
        for i in range(len(col_names)):
            columns.append(Column(col_names[i], CONFIG["default_lines"][i][0][0],  CONFIG["default_lines"][i+1][0][0]))
        print("Columns are set with default")
        return columns

def _clean_col_res(col, res):

    if col.name in ["Cônes/Seringues pour vérification", "Commentaires", "Modèle"]:
        text_res = []
        for t, text in enumerate(res):
            new_text = ""
            start = 0
            for i, letter in enumerate(text[2:],2):
                if letter.lower() == "l":
                    if text[i-1].isnumeric():
                        new_text += text[start:i] + "µ" #insert
                        start=i
                    elif text[i-1].lower() != "m" and text[i-2].isnumeric():
                        new_text += text[start:i-1] + "µ" #replace
                        start=i
            new_text += text[start:]
            text_res.append(new_text)
        return text_res
    
    if col.name == "Fabricant":
        models_serie = list(pd.read_excel(LISTS, sheet_name="lists", index_col=None).dropna()[col.name])
        for t, text in enumerate(res):
            i_max, max_val = max(enumerate([jaro_distance(mod.lower(), text.lower()) for mod in models_serie]), key=lambda x: x[1])
            if max_val>0.85:
                res[t] = models_serie[i_max]
        return res
    
    if col.name == "Nb. vol à vérifier":
        for t, text in enumerate(res):
            if not text in ["1","2","3"]:
                res[t] = ""
        return res

    if col.name == "N° identification demandeur":
        for t, text in enumerate(res):
            if "P!P" in text:
                res[t] = text.replace("P!P", "PIP")
            if "P:P" in text:
                res[t] = text.replace("P:P", "PIP")
        return res
    
    else: 
        return res
        
def generate_df(rows, columns):

    def _process_check(row, col):
        mid_x = (col.x_left+col.x_right)*0.5
        checkboxes = row.checkboxes
        if checkboxes == []:
            return ""
        min_cb, min_dist = sorted(list(map(lambda c : (c[0], abs(mid_x-c[0]["BOTTOM_RIGHT_X"])), checkboxes)), key=lambda t: t[1])[0]

        if min_dist>(mid_x - col.x_left)*0.75: # To far from the 3/4 of the col
            return ""
        else :
            mid_cb = (min_cb["BOTTOM_RIGHT_Y"] + min_cb["TOP_LEFT_Y"])*0.5
            if abs(mid_cb-row.y_top)<abs(mid_cb-row.y_bot):
                return CONFIG["check_columns"][col.name][0]
            else:
                return CONFIG["check_columns"][col.name][1]
    
    def _get_cell(col, row):
        if col.name in list(CONFIG["check_columns"].keys()):
            return _process_check(row, col)
        else :
            for text in row.row_text:
                if text["box"][0]<=mid_x<=text["box"][2]:
                    return text["text"].strip(" ")
            return ""        
    
    # Handle cols from the scan
    rows = [r for r in rows if not r.header]
    res_dict = {}
    for col in columns:
        mid_x = (col.x_left+col.x_right)*0.5
        col_res = []
        for row in rows:
            col_res.append(_get_cell(col, row))
        res_dict[col.name] = _clean_col_res(col, col_res)
    
    # Handle added cols
    for col_name in CONFIG["added_columns"].keys():
        col_res = []

        for row in rows:

            # Add the duplicate info
            if col_name=="LIGNE DUPLIQUEE":
                col_res.append(str(row.duplicate))

        res_dict[col_name] = col_res

    order_df = pd.DataFrame(res_dict, columns=list(CONFIG["columns"].keys())+list(CONFIG["added_columns"].keys()))

    return order_df

def add_new_order(new_df, output_xlsx, sheet="Enregistrement", start_check=1):

    wb = openpyxl.load_workbook(output_xlsx, read_only=False)
    sheet = wb["Enregistrement"]
    max_row = start_check
    for max_row, row in enumerate(list(sheet.rows)[start_check:], start_check): # Last wrote row
        if not any([c.value for c in row[0:10]]): # Check if fields A,B, C are empty or not
            break
        
    for i in range(len(new_df)):
        for col_df, col_xls in CONFIG["columns"].items():
            if col_xls:
                sheet[col_xls+str(i+max_row+1)].value = new_df[col_df].iloc[i]

        for col_df, col_xls in CONFIG["added_columns"].items():
            if col_xls:
                sheet[col_xls+str(i+max_row+1)].value = new_df[col_df].iloc[i]

    wb.save(output_xlsx)

def Tool(pdf_path, output_path):
    """The main function that link all steps

    Args:
        path (_type_): _description_

    Returns:
        _type_: _description_
    """

    scan_name = os.path.splitext(os.path.basename(pdf_path))[0]

    print(f"-- START : Les commandes du scan '{scan_name}' vont être analysées -- ")
    start = time.time()
    # Get the two images of the order
    images = images_from_PDF(pdf_path, rot=True)
    client_image = process_image(images[0])

    raw_order_images = images[1::2] # Non binarized image

    # order_images = [client_image]
    for i_order, raw_order_image in enumerate(raw_order_images):

        # Otsu binaraized is the main image to use
        order_image = process_image(raw_order_image)
        # plt.imshow(order_image)
        # plt.show()
        y_im, x_im = order_image.shape[:2]

        # First process the image to extract structural elements
        checkboxes, lines = get_structural_elements(order_image, raw_order_image, y_im, show=False)

        # plt.imsave(r"check.png", order_image, cmap="gray")

        # Now read the image using PaddleOCR
        lineless_image = delete_HoughLines(order_image, lines["horizontal"])
        raw_texts = paddle_OCR(order_image, show=False)

        # Once it's done, create Line object wich resume texts, checkboxes, and lines
        rows = get_text_lines(lines, checkboxes, raw_texts, x_im, y_im, delta=20)
        columns = get_columns(lines, rows)

        # Finally use columns and lines informations to set a results dataframe
        order_df = generate_df(rows, columns)

        # Save the result
        # order_df.to_excel(output_path, index=False)
        add_new_order(order_df, output_xlsx=output_path)

        print(f"-> Scan {os.path.basename(pdf_path)} page {i_order}: {len([1 for r in rows if not r.header])} commandes détéctées et ajoutées.\n   (Deux type de verifications = deux lignes)")
        
        return order_df

def main(input_path, config=CONFIG):
    # Given path is a pdf
    
    output_path = config["output_path"]

    if os.path.splitext(input_path)[-1].lower() == ".pdf":
        Tool(input_path, output_path)

    # Give path is a folder containing pdf
    if os.path.isdir(input_path):
        all_pdf_in_dir = get_all_pdfs_pathes(input_path)
        for pdf_path in all_pdf_in_dir:
            Tool(pdf_path, output_path)

if __name__ == "__main__":

    # A EFFACER SI ON VEUT JUSTE LANCER L'OUTIL
    print("####### START #######")

    print("\nLancement de l'outil !\nAppuyez sur ctrl+c pour interrompre")
    
    import os

    start = time.time()

    # path = input(f"Rentrez le chemin d'accès au pdf : ")
    input_path = os.path.normpath(CONFIG["input_path"]) if os.path.exists(CONFIG["input_path"]) else ""
    copy_path = os.path.normpath(CONFIG["copy_path"]) if CONFIG["copy_path"] else ""

    # Run the tool for each image of each pdf
    if input_path:
        main(input_path, config=CONFIG)
        taken_time = time.time() - start
        print("Fin de la lecture ; Temps - ", round(taken_time,2), "secondes")

    # Save all pdf in the copy folder
    if copy_path and input_path:
        move_pdf(input_path, copy_path)
        print("Les pdfs ont été copiés")
    
    print("\n######### FIN #########")

    time.sleep(8)
