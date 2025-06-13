import matplotlib.pyplot as plt
import numpy as np
import os

def unpickle(file):
    """
    Helper function to unpickle CIFAR-10 data files.
    """
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def save_cifar10_images(num_images=10, data_dir='cifar-10-batches-py', output_dir='sampleImages'):
    """
    Loads a batch from CIFAR-10 and saves a specified number of images as JPG.

    Args:
        num_images (int): The number of images to save.
        data_dir (str): The directory where your CIFAR-10 data batches are located.
                        (e.g., 'cifar-10-batches-py' if you downloaded and extracted it)
        output_dir (str): The directory where the JPG images will be saved.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Path to one of the data batches (e.g., data_batch_1)
    # You might need to change this if your data files are named differently
    # or if you want to pick from a different batch.
    batch_file = os.path.join(data_dir, 'data_batch_1')

    if not os.path.exists(batch_file):
        print(f"Error: CIFAR-10 batch file not found at '{batch_file}'.")
        print("Please make sure you have downloaded and extracted the CIFAR-10 dataset.")
        print("You can usually download it from: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz")
        print(f"Once extracted, ensure 'data_batch_1' (and others) are in the '{data_dir}' folder.")
        return

    print(f"Loading data from: {batch_file}")
    data_batch = unpickle(batch_file)

    # CIFAR-10 images are stored as a 1D array of 3072 bytes (1024 red, 1024 green, 1024 blue)
    # Reshape them into (32, 32, 3) for plotting
    images = data_batch[b'data']
    labels = data_batch[b'labels']

    # CIFAR-10 class names (useful for naming files)
    meta_file = os.path.join(data_dir, 'batches.meta')
    if os.path.exists(meta_file):
        meta = unpickle(meta_file)
        class_names = [name.decode('utf-8') for name in meta[b'label_names']]
    else:
        print(f"Warning: '{meta_file}' not found. Using generic class names.")
        class_names = [f"class_{i}" for i in range(10)]


    print(f"Saving {min(num_images, len(images))} images...")
    for i in range(min(num_images, len(images))):
        # Reshape and reorder channels for matplotlib (HWC format)
        # CIFAR-10 stores data as (R1, ..., Rn, G1, ..., Gn, B1, ..., Bn)
        image_data = images[i].reshape((3, 32, 32)).transpose(1, 2, 0)
        
        # Get the label and class name
        label = labels[i]
        class_name = class_names[label]

        # Save the image
        filename = os.path.join(output_dir, f'cifar10_image_{i}_label_{label}_{class_name}.jpg')
        plt.imsave(filename, image_data)
        print(f"Saved: {filename}")

    print(f"\nSuccessfully saved {min(num_images, len(images))} images to the '{output_dir}' directory.")

if __name__ == "__main__":
    # Make sure you have downloaded and extracted the CIFAR-10 dataset.
    # The 'cifar-10-python.tar.gz' file, when extracted, usually creates
    # a folder named 'cifar-10-batches-py' containing data_batch_1, etc.
    # Set `cifar10_data_path` to wherever that folder is located.
    cifar10_data_path = r"Images\cifar-10-batches-py"
    
    # You can change the number of images to save here
    images_to_save = 10 
    
    save_cifar10_images(num_images=images_to_save, data_dir=cifar10_data_path)