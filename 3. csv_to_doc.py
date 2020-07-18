#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 20:38:40 2020

@author: acceleriser

Description:
Process PCR cvs into txt
"""

import os
import re
import numpy as np
import pandas as pd
import csv
from timeit import default_timer as timer

#change working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print('Current working directory:', os.getcwd())

# Parallel processing, asynchronous
import multiprocessing
processors = multiprocessing.cpu_count() 
print('processors: ', processors)


# Get a list of all of the pdf files in the directory "example_data_PDF"
pdf_dir = "./pdf/"
img_dir = "./img/"
img_pro_dir = "./img_pro/"
csv_dir = "./csv/"
doc_dir = "./doc/"


# %% Load saved data
data = pd.read_csv(csv_dir + 'data.csv', header=0, index_col=0 )
print(data)

# %%
def full_aggregate_sentences_over_lines(dat):
    """
    Aggregates all text marked as being in the same line.  Then finds
    text that was split over multiple lines by checking if the line
    starts with a capital letter or not.
    """
    
    if False:
        # Drop empty entries with no text
        dat = dat[dat['text'].isnull() == False]
        dat = dat[dat['numerical'].isnull()]
    
    dat_group = dat.groupby(["csv_num", "block_num",  "par_num", "line_num"])
    
    # Create aggregate line text
    line_text = dat_group['text'].apply(lambda x: " ".join([str(e) for e in list(x)]).strip("nan "))
    line_text = line_text.reset_index()
    
    # Create line bounding boxes for line groups
    line_text['top'] = dat_group['top'].agg('min').reset_index()['top']
    line_text['bottom'] = dat_group['bottom'].agg('max').reset_index()['bottom']
    line_text['left'] = dat_group['left'].agg('min').reset_index()['left']
    line_text['right'] = dat_group['right'].agg('max').reset_index()['right']
    
    # Identify lines that start with a lowercase letter.  Misses continued
    # lines that start with a number.  If I cared, I would check if the
    # previous line ended with a period.
    line_text['continued_line'] = line_text['text'].apply(lambda x: np.where(re.search("^[a-z].*", x.strip()), True, False))
    
    # Find the sentences that start with a lowercase letter
    results = pd.DataFrame()
    
    row_of_interest = line_text.iloc[0,:]
    
    # Iterate through and aggregate any lines that are continued
    for index, row in line_text.iterrows():
    
        if (row['continued_line']==True) & (index != 0) :
        
            # If continued line, update values
            row_of_interest['text'] = row_of_interest['text'] + " " + row['text']
            row_of_interest['bottom'] = row['bottom']
            row_of_interest['left'] = min([row_of_interest['left'], row['left']])
            row_of_interest['right'] = max([row_of_interest['right'], row['right']])
    
        else:
            results = results.append(row_of_interest)
            row_of_interest = row
         
    # Drop any now-empty
    results = results[results['text'].apply(lambda x: len(x.strip()) > 0)]
    results = results.drop("continued_line", axis=1)
    
    #results['text'] = results['text'].apply(lambda x: re.sub("[^a-z]+", "", x.lower()))
    
    return results

if False:
    aggregate = full_aggregate_sentences_over_lines(data)
        
    aggregate.to_csv(csv_dir + 'aggregate.csv')


# %%
def aggregate_sentences_over_lines(dat):
    """
    Aggregates all text marked as being in the same line.  Then finds
    text that was split over multiple lines by checking if the line
    starts with a capital letter or not.
    """
    
    # Drop empty entries with no text
    dat = dat[dat['text'].isnull() == False]
    
    dat_group = dat[dat['numerical'].isnull()]
    
    dat_group = dat_group.groupby(["csv_num", "png_path", "block_num",  "par_num", "line_num"])
    
    # Create aggregate line text
    line_text = dat_group['text'].apply(lambda x: " ".join([str(e) for e in list(x)]).strip("nan "))
    line_text = line_text.reset_index()
    
    # Create line bounding boxes for line groups
    line_text['top'] = dat_group['top'].agg('min').reset_index()['top']
    line_text['bottom'] = dat_group['bottom'].agg('max').reset_index()['bottom']
    line_text['left'] = dat_group['left'].agg('min').reset_index()['left']
    line_text['right'] = dat_group['right'].agg('max').reset_index()['right']
    
    # Identify lines that start with a lowercase letter.  Misses continued
    # lines that start with a number.  If I cared, I would check if the
    # previous line ended with a period.
    line_text['continued_line'] = line_text['text'].apply(lambda x: np.where(re.search("^[a-z].*", x.strip()), True, False))
    
    # Find the sentences that start with a lowercase letter
    results = pd.DataFrame()
    
    row_of_interest = line_text.iloc[0,:]
    
    # Iterate through and aggregate any lines that are continued
    for index, row in line_text.iterrows():
    
        if (row['continued_line']==True) & (index != 0) :
        
            # If continued line, update values
            row_of_interest['text'] = row_of_interest['text'] + " " + row['text']
            row_of_interest['bottom'] = row['bottom']
            row_of_interest['left'] = min([row_of_interest['left'], row['left']])
            row_of_interest['right'] = max([row_of_interest['right'], row['right']])
    
        else:
            results = results.append(row_of_interest)
            row_of_interest = row
    
    # Format the text field, stripping any accidentally included numbers
    results['text'] = results['text'].apply(lambda x: re.sub("[^a-z]+", "", x.lower()))
    
    # Drop any now-empty
    results = results[results['text'].apply(lambda x: len(x.strip()) > 0)]
        
    return(results.drop("continued_line", axis=1))

# Create a table with aggregated sentences over lines
tic = timer()
agg_text = aggregate_sentences_over_lines(data)
"""
aggregate = pd.DataFrame()
for csv_num in range(max(data.csv_num) + 1):
    dat = data[data.csv_num == csv_num]
    aggregate = aggregate.append(full_aggregate_sentences_over_lines(dat))
  """  
agg_text.to_csv(csv_dir + 'aggregate_fin.csv')
print("processing time: ", int(timer() - tic))

# %%
def find_balance_sheet_pages(agg_text):
    """
    Through holistic steps, identify pages likely to contain the balance
    sheet.  This includes finding sentences starting
    [abbreviated]*balancesheet, and excluding pages containing
    'notestothefinancialstatements' and 'statementof'.
    """
    
    # Get a list of pages likely to be balance sheets
    BS_page_list = pd.unique(agg_text[agg_text['text'].apply(lambda x: np.where( re.search( "^[abbreviated]*balancesheet", x ), True, False))]['csv_num'])
    
    GBS_page_list = pd.unique(agg_text[agg_text['text'].apply(lambda x: np.where( re.search( "^[group]*balancesheet", x ), True, False))]['csv_num'])
    BS_page_list = np.append(BS_page_list, GBS_page_list)

    GBS_page_list = pd.unique(agg_text[agg_text['text'].apply(lambda x: np.where( re.search( "^[consolidated]*balancesheet", x ), True, False))]['csv_num'])
    BS_page_list = np.append(BS_page_list, GBS_page_list)

    GBS_page_list = pd.unique(agg_text[agg_text['text'].apply(lambda x: np.where( re.search( "^consolidatedstatementoffin", x ), True, False))]['csv_num'])
    BS_page_list = np.append(BS_page_list, GBS_page_list)

    BS_page_list = np.unique(BS_page_list, axis = 0) 
    print(BS_page_list,'\n')
    
    pos_page_list = pd.unique(agg_text[agg_text['text'].apply(lambda x: np.where( re.search( "^statementoffin", x ), True, False))]['csv_num'])
    print(pos_page_list,'\n')
    
    # Filter out any page with the words "notes to the financial statements"
    notes_page_list = pd.unique(agg_text[agg_text['text'].apply(lambda x: "notestothefinancialstatements" in x)]['csv_num'])
    print(notes_page_list,'\n')
    
    # Filter out any page with the words "Statement of changes in equity"
    statement_page_list = pd.unique(agg_text[agg_text['text'].apply(lambda x: "statementofchange" in x)]['csv_num'])
    print(statement_page_list,'\n') 
    
    return( [x for x in BS_page_list if x not in list(notes_page_list) + list(statement_page_list)] + list(pos_page_list) )

csv_numbers = find_balance_sheet_pages(agg_text)
print(csv_numbers)

# %%

# Lifted this almost directly from David Kane's work
def detect_lines(page_df, x_tolerance=0, height_tolerance=20.0):
    """
    Detect lines in the csv of a page, returned by Tesseract
    """
    words_df = page_df[page_df['word_num'] > 0]
    page_stats = page_df.iloc[0, :]
    
    row_ranges = []
    this_range = []
    
    # Clean up the words list, removing blank entries and null values that can arise
    words_df = words_df[words_df['text'].apply(lambda x: str(x).strip() != "")]                   # blank strings
    words_df = words_df[words_df['text'].apply(lambda x: str(x).strip("|") != "")]                # Vertical separators (arise when edges of bounded tables detected)
    words_df = words_df[words_df['text'].isnull() ==False]                                        # Null values (I don't know how they can even exist!)
    words_df = words_df[words_df['height'] < page_stats['height'] / height_tolerance]             # Any word with height greater than 1/20th fraction of page height
    words_df.to_csv(csv_dir + 'debug.csv')
    # Iterate through every vertical pixel position, top (0) to bottom (height)
    for i in range(page_stats['height']): 
        result = (( words_df['bottom'] >= i ) & ( words_df['top'] <= i )).sum() > 0
        
        # Append vertical pixels aligned with words to this_range
        if result:
            this_range.append(i)
        
        # If we've passed out of an "occupied" range, append the resulting range to a list to store
        else:
            if this_range:
                row_ranges.append(this_range)
            this_range = []
            
    # Create bounding boxes for convenience
    return[{"left":0, "right":page_stats['width'], "top":min(r), "bottom":max(r)} for r in row_ranges]


def extract_lines(page_df, lines):
    
    # Look, dark magic!
    finance_regex = r'(.*)\s+(\(?\-?[\,0-9]+\)?)\s+(\(?\-?[\,0-9]+\)?)$'
    
    words_df = page_df[page_df['word_num'] > 0]
    
    raw_lines = []
    results = pd.DataFrame()
    for line in lines:
        
        # Retrieve all text in line
        inline = (words_df['bottom'] <= line['bottom']) & (words_df['top'] >= line['top'])
        line_text = " ".join([str(x) for x in words_df[inline]['text']] )
        
        # Remove any character that isn't a letter, a number or a period
        line_text = re.sub(r'[^a-zA-Z0-9. +]', "", line_text)
        raw_lines.append(line_text)
        
        # Perform an incredibly complex regex search to extract right-most two numbers and the label
        result = re.match(finance_regex, line_text)
        
        # Retrieve the NN's confidence in its translations
        confidences = list(words_df[inline]['conf'])
        
        if result:
            
            try:
                # Check if label is a continuation, if so, append text from last line
                if re.match(r'^[a-z]',re.sub("[0-9]", "", result.groups()[0]).strip()[0]):
                    label = raw_lines[-2] + " " + re.sub("[0-9]", "", result.groups()[0]).strip()
                else:
                    label = re.sub("[0-9]", "", result.groups()[0]).strip()
            
                results = results.append({"label":label,
                                        "value":result.groups()[1],
                                        "currYr":True,
                                        "source":line_text,
                                        "conf":confidences[-1]},
                                        ignore_index=True)
            
                results = results.append({"label":label,
                                        "value":result.groups()[2],
                                        "currYr":False,
                                        "source":line_text,
                                        "conf":confidences[-2]},
                                        ignore_index=True)
            except:
                print("Failed to process line: " + line_text)
    
    return(results)



results = pd.DataFrame()
    
# Iterate through balance sheet pages, retrieve everything
for csv_number in csv_numbers[:1]:
    page_df = data[data['csv_num'] == csv_number]
    
    # Determine where the lines are
    detected_lines = detect_lines(page_df)

    # Get all detectable balance sheet stats
    results = results.append( extract_lines(page_df, detected_lines) )


results.to_csv(csv_dir + 'results.csv')
    
# %%

def determine_units_count(subset_df):
    """
    Simplistic method that finds the units of numbers through counting
    all strings that start with given units, and returning the most common.
    """
    subset = subset_df.copy()
    
    units_regex = "[£$]|million|thousand|£m|£k|$m|$k"
    
    # Search for matches
    subset['keyword_found'] = subset['text'].apply(lambda x: bool(re.match(units_regex, str(x))))
    
    subset = subset[subset['keyword_found']==True]
    subset['count'] = 1
    
    # List and sort units by count
    units=subset[["text", "count"]].\
          groupby("text").\
          count().\
          sort_values("count", ascending=False).\
          reset_index()
    
    # Return most common units
    return( (units.loc[0, "text"], units.loc[0, "count"]) )


def determine_years_count(subset_df, limits=[2000, 2050]):
    """
    Simplistic method that finds the years for a document through
    counting all year-format strings, finding the most common.
    """
    
    subset = subset_df.copy()

    # Search in the value range of interest
    subset['keyword_found'] = (subset['numerical'] >= limits[0]) & (subset['numerical'] <= limits[1])
    
    subset = subset[subset['keyword_found']==True]
    subset['count'] = 1
    
    candidates = subset[["numerical", "count"]].\
                        groupby("numerical").\
                        count().\
                        reset_index().\
                        sort_values("count", ascending=False)['numerical'].values
    
    return(np.array([candidates[0], candidates[0] - 1]))


years = determine_years_count(data)
units = determine_units_count(data)

results['year'] = np.where( results['currYr']==True, years.max(), years.min() )
results['unit'] = units[0]

print(results)
results.to_csv(csv_dir + 'results_full.csv')


