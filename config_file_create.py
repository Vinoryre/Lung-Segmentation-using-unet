from configparser import ConfigParser
'''
    This file is used to generate a configuration file (lung.conf).
    The configuration file defines various path variables and pylidc parameters, making it easier to manage data paths and settings.
    Modify paths and pylidc parameters without altering the code directly.
'''
if __name__ == "__main__":

    config = ConfigParser()

    # Define dataset paths and processing settings
    config['prepare_dataset'] = {
        'LIDC_DICOM_PATH': 'H:/LIDC-IDRI/1_lung_segmentation_test/manifest-1745761260157/LIDC-IDRI', # Absolute path for the LIDC dataset
        'Lung_CT_PATH': './mydata/Image/ct', # Store the original CT images after data processing and numbering
        'Lung_IMAGE_PATH': './mydata/Image/lung', # Stores the lung segementation mask of the corresponding CT image, not binarized
        'Lung_CT_TEST_PATH': './mydata/Image_test/ct',# This is the folder I use to test the trained model to predict the lung segmentation of a patient's CT image. It stores all the slices of the patient's CT.
        'Mask_Threshold':8 # Threshold for mask generation
    }

    # Define pylidc parameters
    config['pylidc'] = {
        'confidence_level': 0.5,
        'padding_size': 512
    }

    # Write to the config file
    with open('lung.conf', 'w') as f:
        config.write(f)

