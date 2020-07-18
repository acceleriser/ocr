#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 20:38:40 2020

@author: acceleriser

Description:
Extract pages from PDFs from pdf sub-folder into images saved to img folder. 
Preprocess images to for OCR.

"""
import cv2
from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#change working directory
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print('Current working directory:', os.getcwd())

# Parallel processing, asynchronous
import multiprocessing
import _multiprocessing_worker_pdf_to_png as mpw
processors = multiprocessing.cpu_count() 
print('processors: ', processors)


# Get a list of all of the pdf files in the directory "example_data_PDF"
pdf_dir = "./pdf/"
img_dir = "./img/"
img_pro_dir = "./img_pro/"
csv_dir = "./csv/"
doc_dir = "./doc/"


def imageplot(image):
    plt.imshow(image, cmap = 'Greys_r')
    plt.show()  
    plt.clf()
    plt.cla()
    plt.close()
    return

# %% Convert PDFs to images, multiprocessing
files = [pdf_dir + filename for filename in os.listdir(pdf_dir) if ".pdf" in filename]
#[print(file) for file in files]

total_files = len(files) + 1
tic = timer()

threads = max(1, int(processors / 2))
print('Threads used: ', threads, ' total_files =', total_files, ' time now ', int(tic))
        
results = pd.DataFrame()
if __name__ == '__main__':
    with multiprocessing.Pool(processes=threads) as pool:
        output = pool.map_async(mpw.worker, files, chunksize=1)

        while not output.ready():
            output.wait(timeout=60)
            tac = timer()
            files_left = output._number_left
            time_est = int(((tac - tic) / (total_files - files_left)) * files_left / 60)
            print("files left: {}".format(files_left), ' minutes left:', time_est)

        pool.close()
        pool.join()
        # append to table
        results = results.append(output.get(), sort=False)

print(results)
print("time for parallel calculation in seconds: ", int(timer() - tic))
print('files processed ', len(output.get()))


# %% Convert PDFs to images, single thread
single_thread = False

if single_thread:
    #read file names from pdf folder
    files = [pdf_dir + filename for filename in os.listdir(pdf_dir) if ".pdf" in filename]
    [print(file) for file in files]
    print(len(files), 'files')
    
    # Process single PDF to multiple png files of individual pages
    tic = timer()
    for file in files:
    
        t0 = timer()
    
        doc = file.split('/')[-1].strip(".pdf")  
    
        os.system("pdftocairo -r 300 -png " + file + " " + img_dir + doc )
            
        print('Converting ', file , ' to images took: ', int(timer()-t0), 'seconds')
    
    print('seconds for single thread calculations took ', int(timer() - tic))

# %% Preprocess all images by applying a number of cleaning steps to them

#find all of the converted pages
img_files = [img_dir + filename for filename in os.listdir(img_dir) if ".png" in filename]
print(len(img_files), 'image files')

print("Pre-processing on all png images (multicore)")

def preprocess(pngname, img_pro_dir = img_pro_dir):
    img_name = pngname.split('/')[-1]
    #print(img_name)
    # Read in as greyscale
    concatenated = cv2.imread(pngname, cv2.IMREAD_GRAYSCALE)
    #imageplot(concatenated)

    # Noise removal
    #concatenated = cv2.medianBlur(concatenated,5)

    # Threshold image to black/white (threshold = 127 so far)
    num, grey_composite = cv2.threshold(concatenated, 127, 255, cv2.THRESH_BINARY)

    # inverting the image for morphological operations
    inverted_composite = 255-grey_composite

    # Perform closing, dilation followed by erosion
    kernel = np.ones((2,2), np.uint8) 
    closed_composite = cv2.morphologyEx(inverted_composite, cv2.MORPH_CLOSE, kernel)

    # Undo inversion
    closed_composite = 255-closed_composite
    #imageplot(closed_composite)

    # Write over original with processed version
    cv2.imwrite(img_pro_dir + img_name, closed_composite)
    
    return pngname

t1 = timer()
for pngname in img_files:
    preprocess(pngname)

print('Pre=processing took: ', int(timer()-t1), 'seconds')