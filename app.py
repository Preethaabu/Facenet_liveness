from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
from classifier import training
import os
from head_position import face_scan
from preprocess import preprocesses
from data_augmentation import augmentation

# Enter the user
name = "Preetha"
# Path for saving the scanned image
folder_path = f'./train_img/{name}'

# Paths for Data preprocessing
input_datadir = './train_img'
output_datadir = './aligned_img'

# Paths for training
datadir = './aligned_img'
modeldir = './model/20180402-114759.pb'
classifier_filename = './class/classifier.pkl'


def main():
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print("Folder created successfully!")
        face_scan(name)
        print("Face scanned successfully")
        # augmentation(name)
        print("Face Augmentation completed")

        obj = preprocesses(input_datadir, output_datadir)
        nrof_images_total, nrof_successfully_aligned = obj.collect_data()

        print('Total number of images: %d' % nrof_images_total)
        print('Number of successfully aligned images: %d' % nrof_successfully_aligned)

        augmentation(name)

        print("Training Start")
        obj = training(datadir, modeldir, classifier_filename)
        get_file = obj.main_train()
        print('Saved classifier model to file "%s"' % get_file)
        sys.exit("All Done")
    else:
        print("Folder already exists cannot scan!")
        obj = preprocesses(input_datadir, output_datadir)
        nrof_images_total, nrof_successfully_aligned = obj.collect_data()

        print('Total number of images: %d' % nrof_images_total)
        print('Number of successfully aligned images: %d' % nrof_successfully_aligned)

        augmentation(name)

        print("Training Start")
        obj = training(datadir, modeldir, classifier_filename)
        get_file = obj.main_train()
        print('Saved classifier model to file "%s"' % get_file)
        sys.exit("All Done")


main()