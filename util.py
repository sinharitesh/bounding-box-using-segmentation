# For distributing images in zipped files, for space constraints.

import os
import zipfile
import math

WORKING_DIR = './data-siim/train/images/'
TOTAL_FILES = len(os.listdir(WORKING_DIR))
FILENAMES = os.listdir(WORKING_DIR)
FILE_BATCH = 1500

for i in range(math.ceil(len(FILENAMES)/FILE_BATCH)):
    print("processing file batch:", i )
    start_index = i* FILE_BATCH
    end_index = ((i+1)* FILE_BATCH)
    end_index = min(end_index, len(FILENAMES))
    temp_files = FILENAMES[start_index: end_index]
    DEST_ZIP_FILE = f'train-images-{start_index}-{end_index-1}.zip'
    print( start_index, end_index, len(temp_files), DEST_ZIP_FILE)
    zipf = zipfile.ZipFile( DEST_ZIP_FILE, 'w', zipfile.ZIP_DEFLATED)
    zipdir('./data-siim/train/images/', zipf, temp_files)
    zipf.close()
# ----------------------------------------------------------------------#    
    
    
    
