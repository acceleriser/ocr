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
import subprocess
from timeit import default_timer as timer
from time import sleep
import csv
from io import StringIO
from multiprocessing import Pool, cpu_count
import pandas as pd
from PIL import Image as im
import numpy as np
import pytesseract

# pylint: disable=C0103

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # change directory
print(os.getcwd(), 'is the current working directory',)

import _pdf_to_doc_functions as fn  # support functions

processors = cpu_count()
print(processors, 'processors')

# Get a list of all of the pdf files in the directory "example_data_PDF"
home_dir = os.path.expanduser("~") + '/Data/PDF/'
pdf_dir = home_dir + "Original/"
pdf_repaired_dir = home_dir + "Repaired/"
img_dir = home_dir + "Img/"
img_pro_dir = home_dir + "Img_pro/"
csv_dir = home_dir + "csv/"
doc_dir = home_dir + "doc/"

# read file names from pdf folder
files = [pdf_dir + filename for filename in os.listdir(pdf_dir) 
         if ".pdf" in filename]
total_files = len(files)
print(total_files, 'files')

t0 = timer()

# Execution options
multi_thread = True

start_new = False
clean_dirs = True # if starting new
repair_pdfs = False # for malformed PDFs



# %%
def clean_dir(mydir):
    filelist = [ f for f in os.listdir(mydir) ]
    for f in filelist:
        os.remove(os.path.join(mydir, f))

if start_new and clean_dirs:
    #clean_dir(pdf_dir)
    #clean_dir(pdf_repaired_dir)
    clean_dir(img_dir)
    clean_dir(img_pro_dir)
    #clean_dir(csv_dir)
    clean_dir(doc_dir)

# %% repair malformed pdfs first

def worker0(wfile):  # multi-thread worker
    """PDF repair worker"""
    wretval = fn.repairpdf(wfile, save_dir=pdf_repaired_dir)
    return wretval

if start_new and repair_pdfs:
    print('starting pdf repairs')
    if multi_thread:
        tic = timer()
        threads = max(1, int(processors))
        print(threads, 'threads used', ' total_files =', total_files, ' time now ', int(tic - t0))

        if __name__ == '__main__':
            with Pool(processes=threads) as pool:
                output = pool.map(worker0, files, chunksize=1)
                pool.close()
                pool.join()

        print(output)
        print(int(timer() - tic), "seconds for parallel repairs")

        print('taking a break to allow for pdftocairo to complete')
        sleep(3)
        print('break ended, continuing')
    else: #single thread
        for file in files:
            retval = fn.repairpdf(file, save_dir=pdf_repaired_dir)
        print('repair ended')

    files = [pdf_repaired_dir + filename for filename in os.listdir(pdf_repaired_dir) if ".pdf" in filename]
    total_files = len(files)


# %% Convert PDFs to images
tic = timer()

def worker1(wfile):  # multi-thread worker
    """Process single PDF to multiple png filenames of individual pages"""
    wdoc = wfile.split('/')[-1].strip(".pdf")
    print(wdoc + ' | ', end=' ')
    wretval = subprocess.call("pdftocairo -r 300 -png " + wfile + " " + img_dir + wdoc, shell=True)
    return wretval

# Process single PDF to multiple png files of individual pages
if start_new:
    if multi_thread:
        tic = timer()
        threads = max(1, int(processors - 1))
        print(threads, 'threads used', ' total_files =', total_files, ' time now ', int(tic - t0))
    
        if __name__ == '__main__':
            with Pool(processes=threads) as pool:
                output = pool.map(worker1, files, chunksize=1)
                pool.close()
                pool.join()
    
            print(output)
            print('\n', int(timer() - tic), "seconds for exploding pdf into images")
    
        print('taking a break to allow pdftocairo complete operations')
        sleep(10)
        print('break ended, continuing')
    else: #single thread
        tac = timer()
        for file in files:
            tic = timer()
            doc = file.split('/')[-1].strip(".pdf")
    
            retval = subprocess.call("pdftocairo -r 300 -png " + file + " " + img_dir + doc, shell=True)
            print(doc, retval)
            print('Converting ', file, 'to images took: ', int(timer()-tic), 'seconds')
    
        print(int(timer() - tac), 'seconds for single thread calculations')


# %% Preprocess all images by applying a number of cleaning steps to them

# find all of the converted pages
img_files = [img_dir + filename for filename in os.listdir(img_dir) if ".png" in filename]
total_files = len(img_files)
print(total_files, 'image files')


def worker2(wfile): # multiprocessing worker
    """Pre-process single image"""
    wretval = fn.preprocess(wfile, save_dir=img_pro_dir)
    print('.', end='')
    return wretval

if start_new:
    img_pro_files = [img_pro_dir + filename for filename in os.listdir(img_pro_dir) if ".png" in filename]
    for f in img_pro_files:
        os.remove(f) # clean results folder
    
    if multi_thread:  # multi-thread processing
        tic = timer()
        threads = max(1, int(processors)) #find the sweet spot for speed
        print(threads, 'threads used', ' total_files =', total_files, ' time now ', int(tic - t0))
    
        if __name__ == '__main__':
            with Pool(processes=threads) as pool:
                output = pool.map(worker2, img_files, chunksize=1)
                pool.close()
                pool.join()
    
        print('\n', int(timer() - tic), "seconds for parallel pre-processing")
        
    else: # single thread processing
        t1 = timer()
        for item in range(total_files):
            fn.preprocess(img_files[item], save_dir=img_pro_dir)
            if item % 100 == 0:
                print(int(item * 100 / total_files), '%', end=" ")
    
        print('\n', int(timer()-t1), 'seconds for single-thread pre-processing')


# %% Images to csv

#read file names from pdf folder
files = [img_pro_dir + filename for filename in os.listdir(img_pro_dir) if ".png" in filename]

#strip path from doc name
docs = [file.split('/')[-1].strip(".pdf") for file in files]
print(len(docs), 'files')

# Apply OCR to every page
def worker3(wargs):
    """Apply OCR to an individual file"""
    print('.', end='')
    w_pngpath, w_csv_num = wargs
    w_image = im.open(w_pngpath)
    w_ocr_obj = pytesseract.image_to_data(w_image)
    w_page_df = pd.read_csv(StringIO(w_ocr_obj),
                            sep='\t',
                            error_bad_lines=True,
                            quoting=csv.QUOTE_NONE,
                            engine='python')
    w_page_df['png_path'] = w_pngpath
    w_page_df['csv_num'] = w_csv_num
    return w_page_df

if start_new:
    if multi_thread: # multi-thread processing
        tic = timer()
        data = pd.DataFrame()
        threads = max(1, int(processors / 3))  # too slow if run at all threads
        print('Threads used: ', threads, ' total_files =', len(files), ' time now ', int(tic - t0))
    
        if __name__ == '__main__':
            with Pool(processes=threads) as pool:
                args = ((files[i], i) for i in range(len(files)))
                output = pool.map(worker3, args, chunksize=1)
                pool.close()
                pool.join()
    
        output = pd.concat(output)
        data = output
        print('\n', int(timer() - tic), "seconds for parallel OCR")
        print(data)
    
    else: # single thread processing
        data = pd.DataFrame()
        csv_num = 0
        tic = timer()
    
        for pngpath in files:
            print(pngpath)
            image = im.open(pngpath)
            #imageplot(image)
    
            ocr_obj = pytesseract.image_to_data(image)
            #print(ocr_obj)
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
    
        print(data)
        print(int(timer() - tic), "seconds for single-thread OCR")

    data = fn.make_measurements(data)
    print(data)
    
    data['numerical'] = fn.convert_to_numeric(data['text'])
    data.to_csv(csv_dir + 'data.csv', sep='\t')
    # print(data)

# %% Start from here if data is saved
###############################################################

datar = pd.read_csv(csv_dir + 'data.csv', header=0,  sep='\t')
print(datar[:3])
#print(data.equals(datar))

max_csv = max(datar.csv_num) + 1
dat_files = [datar[datar.csv_num == csv_num] for csv_num in range(max_csv)]
print(dat_files[:1])

#map csv numbers to file names
csv_map = pd.DataFrame(datar, columns=['csv_num', 'png_path'])
csv_map.sort_values(by=['csv_num', 'png_path'], inplace=True)
csv_map.drop_duplicates(keep='first', inplace=True)

keys = csv_map['csv_num'].values.tolist()
values = csv_map['png_path'].values.tolist()
dictionary = dict(zip(keys, values))

# %% CSV to DOC normal

def worker4(w_dat):
    """aggregate sentences for one file"""
    w_retval = fn.full_aggregate_sentences_over_lines(w_dat)
    print('.', end='')
    return w_retval


if start_new and multi_thread:
    tic = timer()
    threads = max(1, int(processors))
    print(threads, 'threads used', ' total_files =', max_csv, ' time now ', int(tic - t0))

    if __name__ == '__main__':
        with Pool(processes=threads) as pool:
            output = pool.map(worker4, dat_files, chunksize=1)
            pool.close()
            pool.join()

    print('\n', int(timer() - tic), "seconds for parallel aggregation")

    aggregate_full = pd.concat(output)
    
    aggregate_full['png'] = aggregate_full.csv_num.map(dictionary)
    aggregate_full.to_csv(csv_dir + 'aggregate_full.csv', sep='\t')


# %% Create a table with aggregated sentences over lines

def worker5(w_dat):
    """aggregate sentences for one file"""
    w_retval = fn.aggregate_sentences_over_lines(w_dat)
    print('.', end='')
    return w_retval

if start_new:
    if multi_thread:
        tic = timer()
        threads = max(1, int(processors))
        print(threads, 'threads used', ' total_files =', max_csv, ' time now ', int(tic - t0))
    
        if __name__ == '__main__':
            with Pool(processes=threads) as pool:
                output = pool.map(worker5, dat_files, chunksize=1)
                pool.close()
                pool.join()
    
        print('\n', int(timer() - tic), "seconds for parallel aggregation into lines")
    
        aggregate_fin = pd.concat(output)
        aggregate_fin.to_csv(csv_dir + 'aggregate_fin.csv', sep='\t')
    else: # single thread processing
        print(len(dat_files))
        tic = timer()
        agg_text = pd.DataFrame()
        output = []
        i = 0
        for dat in dat_files:
            output.append(fn.aggregate_sentences_over_lines(dat))
            print(len(output), len(output[i]))
            i += 1
    
        aggregate_fin = pd.concat(output)
        aggregate_fin.to_csv(csv_dir + 'aggregate_fin_single.csv', sep='\t')
        print("processing time: ", int(timer() - tic))
else:
    aggregate_fin = pd.read_csv(csv_dir + 'aggregate_fin.csv', header=0,  
                                sep='\t')

# %%
csv_numbers = fn.find_balance_sheet_pages(aggregate_fin)

csv_numbers = pd.Series(csv_numbers)
#print(csv_numbers)

bs_map = pd.concat([csv_numbers, csv_numbers.map(dictionary)], axis=1)
print(bs_map)
bs_map.to_csv(csv_dir + 'bs_map.csv')


# %%
results = pd.DataFrame()
# Iterate through balance sheet pages, retrieve everything

page_dfs_ = [datar[datar.csv_num == csv_num] for csv_num in csv_numbers]

def worker6(csv_num):
    """retrieve financial indicators"""
    # Determine where the lines are
    page_df_local = datar[datar['csv_num'] == csv_num].copy()
    detected_lines = fn.detect_lines(page_df_local)

    # Get all detectable balance sheet stats
    detectable_stats = fn.extract_lines(page_df_local, detected_lines) #

    #add file ID
    doc = dictionary[csv_num].split('/')[-1].strip(".pdf")
    #print(doc)
    detectable_stats['png'] = doc
    detectable_stats['csv_num'] = csv_num
    
    print('.', end='')
    return detectable_stats

if multi_thread:
    tic = timer()
    threads = max(1, int(processors))
    print(threads, 'threads used', ' total_files =', len(csv_numbers), ' time now ', int(tic - t0))

    if __name__ == '__main__':
        with Pool(processes=threads) as pool:
            output = pool.map(worker6, csv_numbers, chunksize=1)
            pool.close()
            pool.join()

    print('\n', int(timer() - tic), "seconds for parallel aggregation into lines")

    output = pd.concat(output)
    output.to_csv(csv_dir + 'results.csv')
    results = output.copy()

else: # single thread processing
    for csv_num in csv_numbers:
        page_df = datar[datar['csv_num'] == csv_num]
    
        # Determine where the lines are
        detected_lines = fn.detect_lines(page_df)
    
        # Get all detectable balance sheet stats
        detectable_stats = fn.extract_lines(page_df, detected_lines) #
    
        #add file ID
        doc = dictionary[csv_num].split('/')[-1].strip(".pdf")
        print(doc)
        detectable_stats['png'] = doc
        detectable_stats['csv_num'] = csv_num
    
        results = results.append(detectable_stats)

    results.to_csv(csv_dir + 'results.csv')

# we are here

# %%
full_results = pd.DataFrame()

for csv_num in csv_numbers:

    page_df = results[results['csv_num'] == csv_num].copy()

    subset_df = datar[datar['csv_num'] == csv_num]

    try:
        years = fn.determine_years_count(subset_df)
    except: # pylint: disable=bare-except
        years = np.array([0, 1])
        #traceback.print_exc()

    try:
        units = fn.determine_units_count(subset_df)
    except: # pylint: disable=bare-except
        units = '???'
        #traceback.print_exc()

    print(dictionary[csv_num], years, units)

    page_df['year'] = np.where(page_df['currYr'] == True, years.max(), years.min())
    page_df['unit'] = units[0]

    full_results = full_results.append(page_df)

#results['year'] = np.where(results['currYr']==True, years.max(), years.min())
#results['unit'] = units[0]

print(full_results)
full_results.to_csv(csv_dir + 'results_full.csv')
print('time now ', timer() - t0)
