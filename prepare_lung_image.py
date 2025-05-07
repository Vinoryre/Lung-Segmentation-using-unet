from utils import is_dir_path, segment_lung
import os
from pathlib import Path
from configparser import ConfigParser
import warnings
import pylidc as pl
from tqdm import tqdm
import numpy as np

'''
    This file uses pylidc to preprocess the LIDC dataset downloaded from the official website, 
    so that the patient data we want is placed in the corresponding folder in the specified format, 
    which is convenient for subsequent model training input. 
    You will get a file directory similar to the following:
    mydata/
        image/
            ct/
                LIDC-IDRI-0001/
                    0001_slice000.npy
                    ...
            lung/
                LIDC-IDRI-0001/
                    0001_slice000.npy
                    ...
    The folder ct contains the original CT images of the patient, 
    and the folder lung contains the corresponding lung segmentation of the patient, 
    without binarization.
'''

# Ignore all warnings
warnings.filterwarnings(action='ignore')

# Read the configuration file
parser = ConfigParser()
parser.read('lung.conf')

# Get paths and parameters from the configuration file
DICOM_DIR = is_dir_path(parser.get('prepare_dataset', 'LIDC_DICOM_PATH')) # Path for LIDC dataset
LUNG_IMAGE_DIR = is_dir_path(parser.get('prepare_dataset', 'LUNG_IMAGE_PATH')) # Path to store lung segmentation images
LUNG_CT_DIR = is_dir_path(parser.get('prepare_dataset', 'LUNG_CT_PATH')) # Path to store original CT images

# Get mask threshold and pylidc parameters
mask_threshold = parser.getint('prepare_dataset', 'Mask_Threshold')

confidence_level = parser.getfloat('pylidc','confidence_level')
padding = parser.getint('pylidc', 'padding_size')

# Class for preparing the ct/lung dataset
class MakeDataSet:
    def __init__(self, LIDC_Patients_list, LUNG_IMAGE_DIR, LUNG_CT_DIR, mask_threshold, padding, confidence_level):
        # Initialization, setting the paths and parameters needed for dataset preparation
        self.IDRI_list = LIDC_Patients_list # List of patients in the LIDC dataset
        self.lung_image_path = LUNG_IMAGE_DIR # Path to store lung images
        self.lung_ct_path = LUNG_CT_DIR # Path to store CT images
        self.mask_threshold = mask_threshold # Mask threshold for segmentation
        self.c_level = confidence_level # Confidence level parameter for pylidc
        self.padding = [(padding, padding), (padding, padding), (0, 0)] # Padding for image processing

    def prepare_dataset(self):
        # Prepare the dataset by processing CT images and generating segmentation masks
        prefix = [str(x).zfill(3) for x in range(1000)] # Assign unique numbers to slices (e.g., 001, 002, etc.)

        LUNG_IMAGE_DIR = Path(self.lung_image_path) # Convert to Path object
        LUNG_CT_DIR = Path(self.lung_ct_path) # Convert to Path object

        for patient in tqdm(self.IDRI_list):
            pid = patient # Get the patient ID
            scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first() # Get the Scanobject for the patient
            vol = scan.to_volume() # Convert the Scan object to a 3D volume (CT image)

            print("Patient ID: {} Dicom Shape: {}".format(pid, vol.shape)) # Print patient ID and image shape

            # Create directories for storing images and masks for this patient
            patient_lung_image_dir = LUNG_IMAGE_DIR / pid
            patient_lung_ct_dir = LUNG_CT_DIR / pid
            Path(patient_lung_image_dir).mkdir(parents=True, exist_ok=True) # Create Lung image directory
            Path(patient_lung_ct_dir).mkdir(parents=True, exist_ok=True) # Create CT image directory

            # Select the middle slices of the CT image
            num_slices = vol.shape[2] # Get the total number of slices in the 3D Volume
            mid = num_slices // 2 # Calculate the index of the middle slice
            selected_slices = range(mid - 5, mid + 5) # Select 5 slices before and after the middle slice, total ten slices

            # Loop through each selected slice
            for slice_idx in selected_slices:
                idx = slice_idx - (mid - 5) # Calculate the offset relative to the middle slice
                # Perform lung segmentation on the slice
                lung_segmented_up_array = segment_lung(vol[:, :, slice_idx])
                lung_segmented_up_array[lung_segmented_up_array == -0] = 0 # Replace -0 values with 0

                # Generate the filename for the slice
                slice_idx_name = "{}_slice{}".format(pid[-4:], prefix[idx])
                # Save the segmented lung image and original CT image
                np.save(patient_lung_image_dir / slice_idx_name, lung_segmented_up_array)
                np.save(patient_lung_ct_dir / slice_idx_name, vol[:, :, slice_idx])


# Main program entry point
if __name__ == '__main__':

    # Get the list of patient from the LIDC dataset
    LIDC_IDRI_list = [f for f in os.listdir(DICOM_DIR)
                      if os.path.isdir(os.path.join(DICOM_DIR, f)) and not f.startswith('.')]
    LIDC_IDRI_list.sort() # Sort the patient list

    # Create a MakeDataSet object and call the prepare_dataset method to prepare the data
    test = MakeDataSet(LIDC_IDRI_list, LUNG_IMAGE_DIR, LUNG_CT_DIR,  mask_threshold, padding, confidence_level)
    test.prepare_dataset()
