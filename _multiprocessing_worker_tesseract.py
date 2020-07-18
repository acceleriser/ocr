# Worker to OCR 
import pytesseract
from PIL import Image as im
import pandas as pd
from io import StringIO
import csv

pdf_dir = "./pdf/"
img_dir = "./img/"

def worker(args):
    # Apply OCR to an individual file
    pngpath, csv_num = args
    image = im.open(pngpath)
    
    ocr_obj = pytesseract.image_to_data(image)
    
    page_df = pd.read_csv(StringIO(ocr_obj),
                            sep='\t',
                            error_bad_lines=True,
                            quoting=csv.QUOTE_NONE,
                            engine='python')

    page_df['csv_num'] = csv_num
    page_df['png_path'] = pngpath

    return page_df



