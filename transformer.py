import os
import cv2
import numpy as np

MAX_IMAGE_RESOLUTION = 1024

last_transform_index = 0


def extract_mask_channel(pixel):
    return np.array([1.0, 1.0, 1.0]) if pixel[1] == 0 and pixel[0] == 255 else np.array([0.0, 0.0, 0.0])


def read_sample(sample):
    image = cv2.imread(sample["image"])
    mask = np.apply_along_axis(extract_mask_channel, 2,
                               cv2.imread(sample["mask"]))

    return {"image": image, "mask": mask}


def scale(sample, dsize):
    image = cv2.resize(sample["image"], dsize=dsize)
    mask = cv2.resize(sample["mask"], dsize=dsize)
    return {"image": image, "mask": mask}


def flip(sample):
    image = cv2.flip(sample["image"].copy(), 1)
    mask = cv2.flip(sample["mask"].copy(), 1)
    return {"image": image, "mask": mask}


def save(output_root,  sample):
    global last_transform_index
    cv2.imwrite(os.path.join(
        output_root, ".".join(["image", str(last_transform_index), "png"])), sample["image"])
    cv2.imwrite(os.path.join(
        output_root, ".".join(["mask", str(last_transform_index),  "png"])), sample["mask"])
    last_transform_index += 1


def main():
    source_root = "./Sneaky"
    output_root = "./Transformed"

    image_root = os.path.join(source_root, "Images")
    mask_root = os.path.join(source_root, "Masks")

    image_filepaths = os.listdir(image_root)

    images = map(lambda image: cv2.imread(
        os.path.join(image_root, image)).shape[:3], image_filepaths)

    images = max(images)

    width = min(int((images[0] + 16) / 32) * 32, MAX_IMAGE_RESOLUTION)
    height = min(int((images[1] + 16) / 32) * 32, MAX_IMAGE_RESOLUTION)

    print("Min dimensions: ", width, height)

    images = map(lambda filepath:
                 filepath.split("."), image_filepaths)

    samples = list(map(lambda filename: {'image': ".".join(filename),
                                        'mask': ".".join(filename[:-1]) + ".png"}, images))

    for i, sample in enumerate(samples):
        print("Image", i, sample["image"], sample["mask"])
        try:
            source = read_sample({
                "image": os.path.join(
                    image_root, sample["image"]),
                "mask": os.path.join(mask_root, sample["mask"])})
            source = scale(source, (width, height))

            save(output_root,  source)

            flipped = flip(source)
            save(output_root, flipped)

        except Exception as err:
            print("Failed to process file:",
                  source["image"], source["mask"], err)


if __name__ == "__main__":
    main()
