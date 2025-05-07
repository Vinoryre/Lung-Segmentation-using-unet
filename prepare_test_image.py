from utils import is_dir_path
import os
from pathlib import Path
from configparser import ConfigParser
import warnings
import pylidc as pl
from tqdm import tqdm
import numpy as np

'''
    This file is similar to prepare_lung_image.py, 
    but it does not process the lung segmentation mask. 
    It only processes the patient's CT image. Because for the last trained model, 
    we only need to input the regularized CT image into it, 
    and expect to get a binary mask of the lung segmentation.
'''

warnings.filterwarnings(action='ignore')

parser = ConfigParser()
parser.read('lung.conf')

DICOM_DIR = is_dir_path(parser.get('prepare_dataset', 'LIDC_DICOM_PATH'))
LUNG_CT_TEST_DIR = is_dir_path(parser.get('prepare_dataset', 'LUNG_CT_TEST_PATH'))

mask_threshold = parser.getint('prepare_dataset', 'Mask_Threshold')

confidence_level = parser.getfloat('pylidc','confidence_level')
padding = parser.getint('pylidc', 'padding_size')

class MakeDataSet:
    def __init__(self, LIDC_Patients_list, LUNG_CT_TEST_DIR, mask_threshold, padding, confidence_level):
        self.IDRI_list = LIDC_Patients_list
        self.lung_ct_test_path = LUNG_CT_TEST_DIR
        self.mask_threshold = mask_threshold
        self.c_level = confidence_level
        self.padding = [(padding, padding), (padding, padding), (0, 0)]

    def prepare_dataset(self):
        prefix = [str(x).zfill(3) for x in range(1000)]

        LUNG_CT_TEST_DIR = Path(self.lung_ct_test_path)

        for patient in tqdm(self.IDRI_list):
            pid = patient
            scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()

            vol = scan.to_volume()
            print("Patient ID: {} Dicom Shape: {}".format(pid, vol.shape))

            patient_ct_test_dir = LUNG_CT_TEST_DIR / pid
            Path(patient_ct_test_dir).mkdir(parents=True, exist_ok=True)

            num_slices = vol.shape[2]
            for slice_idx in range(num_slices):
                slice_idx_name = "{}_slice{}".format(pid[-4:], prefix[slice_idx])
                np.save(patient_ct_test_dir / slice_idx_name, vol[:, :, slice_idx])





if __name__ == '__main__':

    LIDC_IDRI_list = [f for f in os.listdir(DICOM_DIR)
                      if os.path.isdir(os.path.join(DICOM_DIR, f)) and not f.startswith('.')]
    LIDC_IDRI_list.sort()

    test = MakeDataSet(LIDC_IDRI_list, LUNG_CT_TEST_DIR, mask_threshold, padding, confidence_level)
    test.prepare_dataset()