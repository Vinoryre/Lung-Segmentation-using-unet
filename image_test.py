import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

'''
    I didn't write a special function to view the picture, 
    but when I actually process the data, 
    it is still necessary to have a tool to always check what the data in my hand looks like, 
    so this file provides this function
'''

image_file = Path("data/Mask/LIDC-IDRI-0001/0001_NI000_slice000.npy") #Replace the path to the image you want to view here
lung_segmented_up_array = np.load(image_file)

plt.imshow(lung_segmented_up_array, cmap='gray')
plt.title("image:")
plt.show()