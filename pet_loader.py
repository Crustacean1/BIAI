import os
import torch
import numpy as np
import cv2


class PetData:
    def __init__(self, filename):
        self.filename = filename


class PetLoader(torch.utils.data.Dataset):
    def __init__(self, root, mode="train", transform=None):
        assert mode in {"train", "valid", "test"}

        self.root = os.path.join(root, "Oxidized")
        self.mode = mode
        self.transform = transform
        self.filenames = self._find_filenames()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        idx = str(self.filenames[idx])

        img_path = os.path.join(self.root, "image." + str(idx) + ".png")
        mask_path = os.path.join(self.root, "mask." + str(idx) + ".png")

        image = cv2.resize(cv2.imread(img_path),
                           dsize=(256, 256)).transpose(2, 0, 1)
        mask = cv2.resize(cv2.imread(mask_path), dsize=(
            256, 256)).transpose(2, 0, 1)[0]

        max_mask = mask.max()
        mask = mask/max_mask

        return {"image": image, "mask": np.expand_dims(mask, axis=0)}

    def _find_filenames(self):
        image_filenames = os.listdir(self.root)
        image_count = len(
            list(filter(lambda filename: "image" in filename, image_filenames)))

        if self.mode == "train":
            return range(0, int(image_count * 0.8))
        else:
            return range(int(image_count*0.8), image_count)
