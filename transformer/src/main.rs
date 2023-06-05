use image::{imageops, DynamicImage, GenericImageView, ImageBuffer, Rgb, RgbImage};
use rand::Rng;
use std::{
    cmp::min,
    fs, os,
    path::{self, PathBuf},
};

fn retrieve_elements(element_root: PathBuf) -> Result<Vec<PathBuf>, String> {
    let elements = fs::read_dir(&element_root);

    match elements {
        Ok(elements) => Ok(elements
            .filter_map(|image| image.ok().map(|image| image.path()))
            .collect()),
        Err(e) => Err(format!(
            "Failed to read directory '{:?}':\n{}",
            &element_root, e
        )),
    }
}

fn get_samples(image_root: PathBuf, mask_root: PathBuf) -> Result<Vec<(PathBuf, PathBuf)>, String> {
    let images = retrieve_elements(image_root)?;
    let masks = retrieve_elements(mask_root)?;

    Ok(images
        .iter()
        .filter_map(|image| {
            match masks
                .iter()
                .find(|mask| mask.file_stem() == image.file_stem())
            {
                Some(mask) => Some((image.clone(), mask.clone())),
                None => {
                    println!("Image '{:?}' has no corresponding mask", image);
                    None
                }
            }
        })
        .collect())
}

fn convert_mask(mask: DynamicImage) -> DynamicImage {
    let mut image = mask.to_rgb8();
    image.enumerate_pixels_mut().for_each(|(_, _, pixel)| {
        if pixel[0] == 0 && pixel[2] == 255 {
            *pixel = image::Rgb([255, 255, 255])
        } else {
            *pixel = image::Rgb([0, 0, 0])
        }
    });
    DynamicImage::ImageRgb8(image)
}

fn flip(image: &DynamicImage) -> DynamicImage {
    image.fliph()
}

fn blur(image: &DynamicImage) -> DynamicImage {
    DynamicImage::ImageRgb8(imageops::blur(&image.to_rgb8(), 2.))
}

fn noise(image: &DynamicImage) -> DynamicImage {
    let mut image = image.to_rgb8();
    let mut rng = rand::thread_rng();

    image.enumerate_pixels_mut().for_each(|(_, _, pixel)| {
        *pixel = image::Rgb([
            (pixel[0] / 8) * 7 + rng.gen_range(0..32),
            (pixel[1] / 8) * 7 + rng.gen_range(0..32),
            (pixel[2] / 8) * 7 + rng.gen_range(0..32),
        ]);
    });
    DynamicImage::ImageRgb8(image)
}

fn identity(image: &DynamicImage) -> DynamicImage {
    image.clone()
}

fn process_sample(
    sample: &(PathBuf, PathBuf),
) -> Result<Vec<(DynamicImage, DynamicImage)>, String> {
    let (image, mask) = read_sample(sample)?;
    let mask = convert_mask(mask).resize_exact(1024, 1024, imageops::FilterType::Gaussian);
    let image = image.resize_exact(1024, 1024, imageops::FilterType::Gaussian);

    Ok(vec![
        (flip(&image), flip(&mask)),
        (blur(&image), blur(&mask)),
        (noise(&image), identity(&mask)),
        (image, mask),
    ])
}

fn read_sample((image, mask): &(PathBuf, PathBuf)) -> Result<(DynamicImage, DynamicImage), String> {
    let Ok(image) = image::open(image) else {return Err(format!("Failed to read image '{:?}'", image))};
    let Ok(mask) = image::open(mask) else {return Err(format!("Failed to read mask '{:?}'", mask))};
    Ok((image, mask))
}

fn save_sample(
    (image, mask): (DynamicImage, DynamicImage),
    root: &PathBuf,
    i: usize,
) -> Result<(), String> {
    image
        .save(root.join(format!("image.{}.png", i)))
        .map_err(|e| format!("Failed to save image at '{:?}':\n{}", root, e))?;

    mask.save(root.join(format!("mask.{}.png", i)))
        .map_err(|e| format!("Failed to save mask at '{:?}':\n{}", root, e))?;
    Ok(())
}

fn main() -> Result<(), String> {
    let image_root = PathBuf::from("../Pets/Sneaky/Images/");
    let mask_root = PathBuf::from("../Pets/Sneaky/Masks/");
    let transform_root = PathBuf::from("../Pets/Oxidized");

    let samples = get_samples(image_root, mask_root)?;

    let sample_count = samples.len();

    samples
        .iter()
        .enumerate()
        .filter_map(|(i, sample)| match process_sample(sample) {
            Ok(transforms) => {
                println!("Processing: {} / {}", i, sample_count);
                Some(transforms)
            }
            Err(e) => {
                println!(
                    "Image processing failed for: {:?} and {:?}:\n{}",
                    sample.0, sample.1, e
                );
                None
            }
        })
        .flatten()
        .enumerate()
        .for_each(
            |(i, sample)| match save_sample(sample, &transform_root, i) {
                Ok(_) => {}
                Err(e) => println!("Failed to save sample: {}", e),
            },
        );

    Ok(())
}
