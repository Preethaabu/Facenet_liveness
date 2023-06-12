import os
import cv2
import numpy as np
import random

def augment_image(image, darkness):
    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Split HSV image into H, S, V channels
    h_channel, s_channel, v_channel = cv2.split(hsv_image)

    # Apply intensity augmentation to V channel
    v_channel = augment_intensity(v_channel, darkness)

    # Merge the original H, S channels with augmented V channel
    augmented_hsv_image = cv2.merge([h_channel, s_channel, v_channel])

    # Convert HSV image back to BGR color space
    augmented_image = cv2.cvtColor(augmented_hsv_image, cv2.COLOR_HSV2BGR)

    return augmented_image

def augment_intensity(channel, darkness):
    # Apply darkness adjustment
    augmented_channel = cv2.multiply(channel, darkness)

    # Clip pixel values to the valid range
    augmented_channel = np.clip(augmented_channel, 0, 255)

    return augmented_channel.astype(np.uint8)

def augment_flip(image):
    # Flip the image horizontally
    flipped_image = cv2.flip(image, 1)

    return flipped_image

def augment_rotation(image):
    # Generate a random rotation angle between -10 and 10 degrees
    angle = random.uniform(-10, 10)

    # Get the image dimensions
    height, width = image.shape[:2]

    # Calculate the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)

    # Apply rotation to the image
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    return rotated_image

def augmentation(name):
    # Input folder path
    folder_path = f'./aligned_img/{name}'

    # Get a list of all image files within the folder
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # Iterate over each image file
    for file_name in image_files:
        # Read the image
        image_path = os.path.join(folder_path, file_name)
        image = cv2.imread(image_path)

        # Generate a unique darkness value for each image
        darkness = np.random.uniform(low=0.2, high=0.6)

        # Apply different augmentation techniques
        augmented_images = []

        # Original image
        augmented_images.append(image)

        # Augment intensity
        augmented_images.append(augment_image(image, darkness))

        # Augment flip
        augmented_images.append(augment_flip(image))

        # Augment rotation
        augmented_images.append(augment_rotation(image))

        # Save the augmented images
        for i, augmented_image in enumerate(augmented_images):
            # Define the output path for augmented image
            output_path = os.path.join(folder_path, f'augmented_{i}_{file_name}')

            # Save the augmented image
            cv2.imwrite(output_path, augmented_image)

    print("Augmentation completed and augmented images saved in the same folder.")

