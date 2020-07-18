#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 20:38:40 2020

@author: acceleriser

Description:
Extract pages from PDFs from pdf sub-folder into images saved to img folder. 
Preprocess images to for OCR.

"""

import os
import pytesseract
from io import StringIO
import pandas as pd
from PIL import Image as im
import matplotlib.pyplot as plt
import csv
from timeit import default_timer as timer

#change working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print('Current working directory:', os.getcwd())

# Parallel processing, asynchronous
import multiprocessing
import _multiprocessing_worker_tesseract as mpw
processors = multiprocessing.cpu_count() 
print('processors: ', processors)


# Get a list of all of the pdf files in the directory "example_data_PDF"
pdf_dir = "./pdf/"
img_dir = "./img/"
img_pro_dir = "./img_pro/"
csv_dir = "./csv/"
doc_dir = "./doc/"

#read file names from pdf folder
files = [img_pro_dir + filename for filename in os.listdir(img_pro_dir) if ".png" in filename]

#strip path from doc name
docs = [file.split('/')[-1].strip(".pdf") for file in files ]
print(len(docs), 'files')

def imageplot(image):
    plt.imshow(image, cmap = 'Greys_r')
    plt.show()  
    plt.clf()
    plt.cla()
    plt.close()
    return


# %% Apply OCR to every page, single thread
single_thread = False

if single_thread:
    data = pd.DataFrame()
    csv_num = 0
    tic = timer()
    
    for pngpath in files:
        print(pngpath)
        image = im.open(pngpath)
        #imageplot(image)
        
        
        ocr_obj = pytesseract.image_to_data(image)
        #print(ocr_obj )
        #ocr2 = pytesseract.image_to_string(image)
        #print(ocr2)
        
        page_df = pd.read_csv(StringIO(ocr_obj),
                                sep='\t',
                                error_bad_lines=True,
                                quoting=csv.QUOTE_NONE,
                                engine='python')
    
        page_df['csv_num'] = csv_num
        page_df['png_path'] = pngpath
        csv_num += 1
    
        data = data.append(page_df)
        #print(page_df)
    
    print(data)
    print("time for single thread calculation in seconds: ", int(timer() - tic))


# %% Apply OCR, multiprocessing
tic = timer()
data = pd.DataFrame()
threads = max(1, int(processors / 2))
print('Threads used: ', threads, ' total_files =', len(files), ' time now ', int(tic))
        
if __name__ == '__main__':
    with multiprocessing.Pool(processes=threads) as pool:
        args = ( (files[i], i) for i in range(len(files)) )
        output = pool.map_async(mpw.worker, args, chunksize=1)

        while not output.ready():
            output.wait(timeout=60)
            tac = timer()
            files_left = output._number_left
            time_est = int(((tac - tic) / (len(files) - files_left)) * files_left / 60)
            print("files left: {}".format(files_left), ' minutes left:', time_est)

        pool.close()
        pool.join()
        
        # append to dataframe
        data = data.append(output.get(), sort=False)

print("time for parallel calculation in seconds: ", int(timer() - tic))
print('files processed ', len(output.get()))
print(data)


# %%

def make_measurements(data):
    """
    Takes the tabulated OCR output data (pandas DF) and interprets to
    create more variables, geometric information on where elements are
    on the page.
    """
    
    data['centre_x'] = data['left'] + ( data['width'] / 2. )
    data['centre_y'] = data['top'] + ( data['height'] / 2. )
    data['right'] = data['left'] + data['width']
    data['bottom'] = data['top'] + data['height']
    data['area'] = data['height'] * data['width']
    
    return( data )

data = make_measurements(data)
print(data)

# %%

# Create numerical variables from text where possible
def convert_to_numeric(series):
    """
    Converts a pandas series object (of strings) to numeric if possible.
    If not possible, will return numpy.nan.
    """
    q_func = lambda x: str(x).replace(",", "").strip("(").strip(")")
    return( pd.to_numeric(series.apply(q_func), errors="coerce") ) 
    # If errors, force process to continue, invalid element returned as numpy.nan
    
data['numerical'] = convert_to_numeric(data['text'])
data.to_csv(csv_dir + 'data.csv')
print(data)

# %% Start from here if data is saved
datar = pd.read_csv(csv_dir + 'data.csv', header=0, index_col=0 )
print(data.equals(datar))

# for debug
#print(datar)
#print(data)
