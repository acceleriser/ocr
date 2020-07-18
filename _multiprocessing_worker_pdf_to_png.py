# pdf to png multiprocessing worker
import subprocess
img_dir = "./img/"

def worker(file):
    # Process single PDF to multiple png files of individual pages
    doc = file.split('/')[-1].strip(".pdf")  
    print(doc + ' | ')
    retval = subprocess.call("pdftocairo -r 300 -png " + file + " " + img_dir + doc , shell = True)
        
    return retval
