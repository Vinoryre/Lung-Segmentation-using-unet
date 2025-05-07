import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from matplotlib import pyplot as plt
from configparser import ConfigParser
from utils import is_dir_path

from Dataset import LungTestDataset
from UNet_nn import DoubleConv, Down, Up, UNet

# Allow duplicate OpenMp library loading (common workaround for MKL/OpenMP conflicts on some systems)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

parser = ConfigParser()
parser.read('lung.conf')

ct_test_root = is_dir_path(parser.get('prepare_dataset', 'LUNG_CT_TEST_PATH')) # CT_test slices path
dataset = LungTestDataset(ct_test_root)

if __name__ == '__main__':
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_channels=1, n_classes=1)
    model.load_state_dict(torch.load('unet_model.pth'))
    model.to(device)
    model.eval()
    idx = 1
    with torch.no_grad():
        for ct, meta_data in test_loader:
            ct = ct.to(device)
            output = model(ct)
            ct_normalized = ct[0].cpu().squeeze(0)
            ct_min = meta_data['ct_min']
            ct_max = meta_data['ct_max']
            ct_vis = ct_normalized * (ct_max[0] - ct_min[0]) + ct_min[0]
            output_vis = torch.sigmoid(output[0].cpu().squeeze(0))

            fig = plt.figure(figsize=(12, 4))
            axs = fig.subplots(1, 2)
            axs[0].imshow(ct_vis, cmap='gray', vmin=-1000, vmax=400)
            axs[0].set_title('CT Slice')
            axs[1].imshow(output_vis, cmap='gray', vmin=0, vmax=1)
            axs[1].set_title('Lung Segmentation')

            for ax in axs:
                ax.axis('off')

            desktop_path = ""  # TODO: Please manually set this path before running
            if not desktop_path:
                raise ValueError(" You must set 'desktop_path' before running this script.")
            
            save_path = os.path.join(desktop_path, f"pred_slice_{idx}.png")
            plt.savefig(save_path)
            plt.close()
            idx += 1
