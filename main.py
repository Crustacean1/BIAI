import os
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pprint import pprint
import torch
from torch.utils.data import DataLoader
from segmentation_models_pytorch.datasets import SimpleOxfordPetDataset

from pet_model import PetModel
from pet_loader import PetLoader

# download data
root = "."
SimpleOxfordPetDataset.download(root)

# init train, val, test sets
train_dataset = PetLoader(root, "train")
valid_dataset = PetLoader(root, "valid")
test_dataset = PetLoader(root, "test")

# It is a good practice to check datasets don`t intersects with each other
print("test:", len(test_dataset))
print("train:", len(train_dataset.filenames))
print("verification:", len(valid_dataset.filenames))
assert set(test_dataset.filenames).isdisjoint(set(train_dataset.filenames))
# assert set(test_dataset.filenames).isdisjoint(set(valid_dataset.filenames))
# assert set(train_dataset.filenames).isdisjoint(set(valid_dataset.filenames))

print(f"Train size: {len(train_dataset)}")
print(f"Valid size: {len(valid_dataset)}")
print(f"Test size: {len(test_dataset)}")

n_cpu = os.cpu_count()

train_dataloader = DataLoader(
    train_dataset, batch_size=16, shuffle=True, num_workers=n_cpu)
valid_dataloader = DataLoader(
    valid_dataset, batch_size=16, shuffle=False, num_workers=n_cpu)
test_dataloader = DataLoader(
    test_dataset, batch_size=16, shuffle=False, num_workers=n_cpu)

for i in range(0, 3):
    sample = train_dataset[i*5]
    image = sample["image"]
    mask = sample["mask"]
    print("Image: ", image.shape[:3], "Mask: ", mask.shape[:3])
    plt.subplot(1, 2, 1)
    plt.imshow(image.transpose(1, 2, 0))
    plt.subplot(1, 2, 2)
    plt.imshow(mask.transpose(1, 2, 0))
    plt.show()

model = PetModel("FPN", "resnet34", encoder_weights='imagenet',
                 in_channels=3, out_classes=1)

trainer = pl.Trainer(
    gpus=1,
    max_epochs=50,
)

trainer.fit(
    model,
    train_dataloaders=train_dataloader,
    val_dataloaders=valid_dataloader,
)

valid_metrics = trainer.validate(
    model, dataloaders=valid_dataloader, verbose=False)
pprint(valid_metrics)

test_metrics = trainer.test(model, dataloaders=test_dataloader, verbose=False)
pprint(test_metrics)

batch = next(iter(test_dataloader))
with torch.no_grad():
    model.eval()
    logits = model(batch["image"])
pr_masks = logits.sigmoid()

for image, gt_mask, pr_mask in zip(batch["image"], batch["mask"], pr_masks):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image.numpy().transpose(1, 2, 0))  # convert CHW -> HWC
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    # just squeeze classes dim, because we have only one class
    plt.imshow(gt_mask.numpy().squeeze())
    plt.title("Ground truth")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    # just squeeze classes dim, because we have only one class
    plt.imshow(pr_mask.numpy().squeeze())
    plt.title("Prediction")
    plt.axis("off")

    plt.show()
