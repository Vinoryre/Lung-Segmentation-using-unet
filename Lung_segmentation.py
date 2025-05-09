import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from matplotlib import pyplot as plt
from configparser import ConfigParser
from utils import is_dir_path
from Dataset import lungSefDataset
from UNet_nn import DoubleConv, Down, Up, UNet

# Load configuration from lung.conf
parser = ConfigParser()
parser.read('lung.conf')
ct_root = is_dir_path(parser.get('prepare_dataset', 'LUNG_CT_PATH')) # CT slices path
lung_root = is_dir_path(parser.get('prepare_dataset', 'LUNG_IMAGE_PATH')) # Corresponding mask path

# Initialize dataset and split into training and validation sets
dataset = lungSefDataset(ct_root, lung_root)
train_size = int(0.9 * len(dataset)) # 90% for training
val_size = len(dataset) - train_size # 10% for validation

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

if __name__ == '__main__':
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)

    # Initialize U-Net model
    model = UNet(n_channels=1, n_classes=1)

    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    Set_save_path = ""  # TODO: Please manually set this path before running
    if not Set_save_path:
        raise ValueError(" You must set 'Set_save_path' before running this script.")

    # Dice Loss function for training
    def dice_loss(pred, target, smooth = 1e-6):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(1, 2, 3))
        union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()

    # Dice Coefficient for evaluation
    def dice_coeff(pred, target, smooth = 1e-6):
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float() # Threshold prediction
        intersection = (pred * target).sum(dim=(1, 2, 3))
        union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = (2. * intersection + smooth) / (union + smooth)
        return dice.mean()

    # Set optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    num_epochs = 20 # Number of training epochs

    # Track losses and metrics for plotting
    train_losses = []
    val_losses = []
    val_dices = []

    # Progress print function
    def print_progress(epoch, num_epochs):
        progress = (epoch + 1) /  num_epochs * 100
        print(f"Epoch [{epoch+1}/{num_epochs}] - Progress: {progress:.2f}%", end="")

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for ct, mask, meta_data in train_loader:
            ct, mask = ct.to(device), mask.to(device)

            optimizer.zero_grad()
            output = model(ct)
            loss = dice_loss(output, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping
            optimizer.step()

            running_loss += loss.item()

        scheduler.step() # Step the LR scheduler

        train_losses.append(running_loss / len(train_loader)) # Avg training loss for this epoch

        print(f"Loss: {running_loss / len(train_loader):.4f}")

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        saved = False   # Save only one prediction visualization per epoch

        with torch.no_grad():
            for ct, mask, meta_data in val_loader:
                ct, mask = ct.to(device), mask.to(device)
                output = model(ct)
                print(f"out_put mask range: {output.min().item():.2f} ~ {output.max().item():.2f}")

                val_loss += dice_loss(output, mask).item()
                val_dice += dice_coeff(output, mask).item()

                if not saved:
                    # Recover original CT values for visualization
                    ct_normalized = ct[0].cpu().squeeze(0)
                    ct_min = meta_data['ct_min']
                    ct_max = meta_data['ct_max']
                    ct_vis = ct_normalized * (ct_max[0] - ct_min[0]) + ct_min[0]

                    # Apply sigmoid to prediction for visualization
                    output_vis = torch.sigmoid(output[0].cpu().squeeze(0))

                    # Plot CT slice and prediction
                    fig= plt.figure(figsize=(12, 4))
                    axs = fig.subplots(1, 2)
                    axs[0].imshow(ct_vis, cmap='gray', vmin=-1000, vmax=400)
                    axs[0].set_title('CT Slice')
                    axs[1].imshow(output_vis, cmap='gray', vmin=0, vmax=1)
                    axs[1].set_title('Lung Segmentation')

                    for ax in axs:
                        ax.axis('off')

                    save_path = os.path.join(Set_save_path, f"pred_epoch_{epoch+1}.png")
                    plt.savefig(save_path)
                    plt.close()
                    saved = True

        # Record validation metrics
        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)

        val_losses.append(avg_val_loss)
        val_dices.append(avg_val_dice)

        print_progress(epoch, num_epochs)
        print(f" | Val Loss: {avg_val_loss:.4f} | Dice Coeff: {avg_val_dice:.4f}")

        # Save model checkpoint
        torch.save(model.state_dict(), 'unet_model.pth')

    # Free up GPU memory
    del model
    torch.cuda.empty_cache()

    # Plot training and validation curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.plot(val_dices, label='Validation Dice Coeff', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Loss / Dice Coeff')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig("training_curve.png")
    plt.show()



