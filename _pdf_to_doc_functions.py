#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 20:38:40 2020

@author: acceleriser

Description:
Convert PDFs into texts.
Functions here.
"""
import traceback
import re
import subprocess
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# pylint: disable=C0301

pdf_dir = "./pdf/"
pdf_repaired_dir = "./pdfr/"
img_dir = "./img/"
img_pro_dir = "./img_pro/"
csv_dir = "./csv/"
doc_dir = "./doc/"


def imageplot(image):
    """plot images"""
    plt.imshow(image, cmap='Greys_r')
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()
    return True

def repairpdf(file, save_dir=pdf_repaired_dir):
    """repair malformed pdfs first"""
    doc = file.split('/')[-1].strip(".pdf")
    retval = subprocess.call("pdftocairo -pdf " + file + " " + save_dir + doc + ".pdf", shell=True)
    print(doc, retval, end=' | ')
    return retval


def preprocess(pngname, save_dir=img_pro_dir):
    """pre-process pdfs"""
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
    kernel = np.ones((2, 2), np.uint8)
    closed_composite = cv2.morphologyEx(inverted_composite, cv2.MORPH_CLOSE, kernel)

    # Undo inversion
    closed_composite = 255-closed_composite
    #imageplot(closed_composite)

    # Write over original with processed version
    cv2.imwrite(save_dir + img_name, closed_composite)

    return pngname


def make_measurements(w_data):
    """
    Takes the tabulated OCR output data (pandas DF) and interprets to
    create more variables, geometric information on where elements are
    on the page.
    """
    w_data['centre_x'] = w_data['left'] + (w_data['width'] / 2.)
    w_data['centre_y'] = w_data['top'] + (w_data['height'] / 2.)
    w_data['right'] = w_data['left'] + w_data['width']
    w_data['bottom'] = w_data['top'] + w_data['height']
    w_data['area'] = w_data['height'] * w_data['width']

    return w_data


# Create numerical variables from text where possible
def convert_to_numeric(series):
    """Converts a pandas series object (of strings) to numeric if possible.
    If not possible, will return numpy.nan."""
    q_func = lambda x: str(x).replace(",", "").strip("(").strip(")")
    return pd.to_numeric(series.apply(q_func), errors="coerce")
    # If errors, force process to continue, invalid element returned as numpy.nan


def aggregate_sentences_over_lines(dat):
    """
    Aggregates all text marked as being in the same line.  Then finds
    text that was split over multiple lines by checking if the line
    starts with a capital letter or not.
    """

    # Drop empty entries with no text
    dat = dat[dat['text'].isnull() == False]

    dat_group = dat[dat['numerical'].isnull()]

    dat_group = dat_group.groupby(["csv_num", "png_path", "block_num", "par_num", "line_num"])

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

    try: #if empty dataframe, no error raised
        row_of_interest = line_text.iloc[0, :]

        # Iterate through and aggregate any lines that are continued
        for index, row in line_text.iterrows():

            if (row['continued_line'] == True) & (index != 0):

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
    except: # pylint: disable=bare-except
        results = line_text
        traceback.print_exc()

    results = results.drop("continued_line", axis=1)

    return results


def full_aggregate_sentences_over_lines(dat):
    """
    Aggregates all text marked as being in the same line.  Then finds
    text that was split over multiple lines by checking if the line
    starts with a capital letter or not.
    """
    drop_text = False
    if drop_text:
        # Drop empty entries with no text
        dat = dat[dat['text'].isnull() == False]
        dat = dat[dat['numerical'].isnull()]

    dat_group = dat.groupby(["csv_num", "block_num", "par_num", "line_num"])

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

    row_of_interest = line_text.iloc[0, :]

    # Iterate through and aggregate any lines that are continued
    for index, row in line_text.iterrows():

        if (row['continued_line'] == True) & (index != 0):

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


# Lifted this almost directly from David Kane's work
def detect_lines(w_page_df, height_tolerance=20.0):
    """
    Detect lines in the csv of a page, returned by Tesseract
    """
    words_df = w_page_df[w_page_df['word_num'] > 0]
    page_stats = w_page_df.iloc[0, :]

    row_ranges = []
    this_range = []

    # Clean up the words list, removing blank entries and null values that can arise
    # blank strings
    words_df = words_df[words_df['text'].apply(lambda x: str(x).strip() != "")]
    # Vertical separators (arise when edges of bounded tables detected)
    words_df = words_df[words_df['text'].apply(lambda x: str(x).strip("|") != "")]
    # Null values (I don't know how they can even exist!)
    words_df = words_df[words_df['text'].isnull() == False]
    # Any word with height greater than 1/20th fraction of page height
    words_df = words_df[words_df['height'] < page_stats['height'] / height_tolerance]
    words_df.to_csv(csv_dir + 'debug.csv')
    # Iterate through every vertical pixel position, top (0) to bottom (height)
    for x in range(page_stats['height']):
        result = ((words_df['bottom'] >= x) & (words_df['top'] <= x)).sum() > 0

        # Append vertical pixels aligned with words to this_range
        if result:
            this_range.append(x)

        # If we've passed out of an "occupied" range, append the resulting range to a list to store
        else:
            if this_range:
                row_ranges.append(this_range)
            this_range = []

    # Create bounding boxes for convenience
    return[{"left":0, "right":page_stats['width'], "top":min(r), "bottom":max(r)} for r in row_ranges]


def extract_lines(w_page_df, lines):
    """extract numbers"""
    # Look, dark magic!
    finance_regex = r'(.*)\s+(\(?\-?[\,0-9]+\)?)\s+(\(?\-?[\,0-9]+\)?)$'

    words_df = w_page_df[w_page_df['word_num'] > 0]

    raw_lines = []
    w_results = pd.DataFrame()
    for line in lines:

        # Retrieve all text in line
        inline = (words_df['bottom'] <= line['bottom']) & (words_df['top'] >= line['top'])
        line_text = " ".join([str(x) for x in words_df[inline]['text']])

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
                if re.match(r'^[a-z]', re.sub("[0-9]", "", result.groups()[0]).strip()[0]):
                    label = raw_lines[-2] + " " + re.sub("[0-9]", "", result.groups()[0]).strip()
                else:
                    label = re.sub("[0-9]", "", result.groups()[0]).strip()

                w_results = w_results.append({"label":label,
                                              "value":result.groups()[1],
                                              "currYr":True,
                                              "source":line_text,
                                              "conf":confidences[-1]},
                                             ignore_index=True)

                w_results = w_results.append({"label":label,
                                              "value":result.groups()[2],
                                              "currYr":False,
                                              "source":line_text,
                                              "conf":confidences[-2]},
                                             ignore_index=True)
            except: # pylint: disable=bare-except
                print("Failed to process line: " + line_text)
                traceback.print_exc()

    return w_results


def determine_units_count(w_subset_df):
    """
    Simplistic method that finds the units of numbers through counting
    all strings that start with given units, and returning the most common.
    """
    subset = w_subset_df.copy()

    units_regex = "[£$]|million|thousand|£m|£k|$m|$k"

    # Search for matches
    subset['keyword_found'] = subset['text'].apply(lambda x: bool(re.match(units_regex, str(x))))

    subset = subset[subset['keyword_found'] == True]
    subset['count'] = 1

    # List and sort units by count
    w_units = subset[["text", "count"]].\
          groupby("text").\
          count().\
          sort_values("count", ascending=False).\
          reset_index()

    # Return most common units
    return (w_units.loc[0, "text"], w_units.loc[0, "count"])


def determine_years_count(w_subset_df, limits=(2000, 2050)):
    """
    Simplistic method that finds the years for a document through
    counting all year-format strings, finding the most common.
    """

    subset = w_subset_df.copy()

    # Search in the value range of interest
    subset['keyword_found'] = (subset['numerical'] >= limits[0]) & (subset['numerical'] <= limits[1])

    subset = subset[subset['keyword_found'] == True]
    subset['count'] = 1

    candidates = subset[["numerical", "count"]].\
                        groupby("numerical").\
                        count().\
                        reset_index().\
                        sort_values("count", ascending=False)['numerical'].values

    return np.array([candidates[0], candidates[0] - 1])


def find_balance_sheet_pages(agg_text):
    """
    Through holistic steps, identify pages likely to contain the balance
    sheet.  This includes finding sentences starting
    [abbreviated]*balancesheet, and excluding pages containing
    'notestothefinancialstatements' and 'statementof'.
    """

    # Get a list of pages likely to be balance sheets
    BS_page_list = pd.unique(agg_text[agg_text['text'].apply(lambda x: np.where(re.search("^[abbreviated]*balancesheet", x), True, False))]['csv_num'])

    GBS_page_list = pd.unique(agg_text[agg_text['text'].apply(lambda x: np.where(re.search("^[group]*balancesheet", x), True, False))]['csv_num'])
    BS_page_list = np.append(BS_page_list, GBS_page_list)

    GBS_page_list = pd.unique(agg_text[agg_text['text'].apply(lambda x: np.where(re.search("^[consolidated]*balancesheet", x), True, False))]['csv_num'])
    BS_page_list = np.append(BS_page_list, GBS_page_list)

    GBS_page_list = pd.unique(agg_text[agg_text['text'].apply(lambda x: np.where(re.search("^consolidatedstatementoffin", x), True, False))]['csv_num'])
    BS_page_list = np.append(BS_page_list, GBS_page_list)

    BS_page_list = np.unique(BS_page_list, axis=0)
    print(BS_page_list, '\n')

    pos_page_list = pd.unique(agg_text[agg_text['text'].apply(lambda x: np.where(re.search("^statementoffin", x), True, False))]['csv_num'])
    print(pos_page_list, '\n')

    # Filter out any page with the words "notes to the financial statements"
    notes_page_list = pd.unique(agg_text[agg_text['text'].apply(lambda x: "notestothefinancialstatements" in x)]['csv_num'])
    print(notes_page_list, '\n')

    # Filter out any page with the words "Statement of changes in equity"
    statement_page_list = pd.unique(agg_text[agg_text['text'].apply(lambda x: "statementofchange" in x)]['csv_num'])
    print(statement_page_list, '\n')

    return([x for x in BS_page_list if x not in list(notes_page_list) + list(statement_page_list)] + list(pos_page_list))
