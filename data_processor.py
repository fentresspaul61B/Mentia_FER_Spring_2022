# Imports for Iterating through files:
import os
import random
from argparse import ArgumentParser

# Imports for Aging Model
import torch
from torchvision import transforms
from gan_module import Generator

# Imports for images:
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image





def reshape_image(image_path:str):
    """
    Arguments:
        image_path (string)
    Returns:
        reshaped image of shape (3,512,512) (array)
    """

    # Here I am reading in the image, resizing the image using PIL because
    # the images are 48x48 squares, which are not multiple of 512x512.
    # Then once they are 512x512 I use numpy reshape to make them 1x512x512
    img = np.array((Image.open(image_path)).resize((512,512))).reshape((1,512,512))

    # Then once the img is shape 1x512x512 I stack the images ontop of eachother
    # using a numpy stack which results in a (3,1,512,512) shape, and then reshape
    # it again using numpy.
    img = np.stack((img,img,img)).reshape((512,512,3))
    return img



def age_image(image_path:str, folder_path:str=None):
    """
    Arguments:
        image_path (string)
    returns:
        aged_face (np.array)
    """

    # This function takes in an image path and a folder path.
    # Then it ages the image, and saves new image into the
    # specified path called folder_path.

    model = Generator(ngf=32, n_residual_blocks=9)
    ckpt = torch.load('pretrained_model/state_dict.pth', map_location='cpu')
    model.load_state_dict(ckpt)

    trans = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    img = reshape_image(image_path)
    img = Image.fromarray(img)
    img = trans(img).unsqueeze(0)
    aged_face = model(img)
    aged_face = (aged_face.squeeze().permute(1, 2, 0).detach().numpy() + 1.0) / 2.0

    # print(np.array(aged_face).shape, type(aged_face))
    # print("Face Succsefuly Aged.")

    return aged_face



def crop_image(image:np.ndarray):
    """
    Arguments:
        image (np.array)
    returns:
        np.array
    """

    # This function crops the image by dividing by 3.

    assert isinstance(image, np.ndarray),  "Image is not array"
    if len(image.shape) == 2: # Image has no color channel
        X, Y = image.shape
    else: # Image has three channels
        X, Y, _ = image.shape
    crop = image[:int(X // 3), :int(Y//3)]
    return crop


def age_and_crop_image(path):
    """
    Arguments:
        path:string
    Returns:
        aged_image:Array
    Function purpose:
        Age and crop the image.
    Algorithm:
        1. Reshape image to fit pre trained aging model.
        2. Age the image using the pretrained again model.
        3. Crop the image because to not return a 3x3 grid.
    """
    # crop_image(age_image(path))
    return crop_image(age_image(path))


import cv2
from PIL import Image

def age_data_set(intial_folder_path:str, destination_folder_path:str):
    """
    Arguments:
        intial_folder_path:string
        destination_folder_path:string
    Returns:
        None
    Function purpose:
        This function takes in a single folder with images inside for
        a single class at a time. For example it will take in all the
        "angry" images from the train folder, then iterate over them.
        and age them 1 by 1.
    Algorithm:
        for each image_path in folder:
            1. Extract image path.
            2. Age the photo.
            3. Create new file path for aged photo.
            4. Change the photo color to black and white.
            5. Save the aged photo into new empty folder.

    """
    # 1. Extract image path.Create list of all image paths
    image_paths = os.listdir(intial_folder_path) # list to iterative over.

    print("Aging process started. ")

    for count, path in enumerate(image_paths):
        try:
            # 2. Age the photo.
            aged = age_and_crop_image(intial_folder_path + "/" + path) * 255
            aged_image = Image.fromarray(aged.astype(np.uint8))

            # 3. Create new file path for aged photo.
            destination = destination_folder_path + "/" + path

            # 4. Change the color of the image to black and white.
            aged_image = aged_image.convert("L")

            # 5. Save the aged photo into new empty folder.
            # cv2.imwrite(filename, aged_image)
            aged_image.save(destination)

            # This code here is to keep track of the process.
            if count % 20 == 0:
                print(str(count) + " Images aged so far.")
        except:
            print("Image Failed To Age. ")


    print("Aging process completed. ")
    return

# Run this code below to age a dataset.

# How this file works:
# 1. Add a pathname to the "data_path" variable.
# 2. Add a pathname to the "destination_path" variable.
# 3. Run the code by running "python3 data_processor.py" In the terminal.

# data_path = "/Users/paulfentress/Desktop/Mentia_Gans_Data.py/images/train/angry"
# destination_path = "/Users/paulfentress/Desktop/Mentia_Gans_Data.py/images_aged/train/angry"
# age_data_set(data_path, destination_path)


# fig, ax = plt.subplots(1, 1, figsize=(10, 5))
# # test_image = "/Users/paulfentress/Desktop/Mentia_Gans_Data.py/267.jpg"
# test_image = "/Users/paulfentress/Desktop/Mentia_Gans_Data.py/images/images/train/angry/0.jpg"
# aged = age_and_crop_image(test_image)
# # aged_PIL = Image.fromarray(aged.astype(np.uint8))
# # aged_PIL.show()
# ax.imshow(aged)
# plt.show()

# https://www.geeksforgeeks.org/how-to-merge-multiple-folders-into-one-folder-using-python/

# 1. Rename Images in disgust by adding a letter, just so the angry and disgust images
#    dont have duplicate names.
# 2. Add all the images from disgust into angry.
# 3. Rename angry to "angry + disgust"
# 4. Delete empty disgust folder.

# 1. Rename Images in disgust by adding a letter, just so the angry and disgust images
#    dont have duplicate names.

def rename_files(folder_path: str, name_str: str):
    """
      Arguments:
        folder_path: (str). path to folder which contains images for a
        specific class.

        name_str: (str). A string to attach to the old file name, to make
        it into a new file name.

      Output:
          None. This function is just editing files.

      Purpose:
          Rename all the image files inside a folder, to be the previous
          name + name_str. The reason to do this, is because we want to join
          to folders together, and do not want duplicate names.

      Algorithm:
          1. Create list of all the file names.
          2. iterate over the files in the files_names.
              For file_name in file_names:
                  rename file
      """

    file_names = os.listdir(folder_path)
    print("Renaming Files.")
    count = 0
    for file_name in file_names:
        try:
            new_name = folder_path + "/" + name_str + file_name
            old_name = folder_path + "/" + file_name
            os.rename(old_name, new_name)
            count += 1
        except:
            print("Cannot Rename File: " + file_name)
    print("FILE NAMING COMPLETE.")
    print("Renamed: " + str(count) + " files.")

# 2. Add all the images from disgust into angry.
import shutil

def move_files(source_dir: str, target_dir: str):
    """
      Arguments:
        source_dir: (str). Path to move files from.

        target_dir: (str). Path to move files into/

      Output:
          None. This function is just editing files.

      Purpose:
          Move image files into the source_dir from the
          target_dir.
      Algorithm:
          1. Create list of all the file names.
          2. iterate over the files in the files_names.
              For file_name in file_names:
                  move file from source_dir into target_dir
    """
    print()
    print("MOVING FILES")

    file_names = os.listdir(source_dir) # list
    target_length = len(os.listdir(target_dir)) # int
    print("Attmepting to move: " + str(len(file_names)) + " files.")
    print("Target Folder has: " + str(target_length) + " files. ")

    files_moved = 0
    for file_name in file_names:
        try:
            shutil.move(os.path.join(source_dir, file_name), target_dir)
            files_moved += 1
        except:
            print("Could not move file: " + file_name)

    print("FILE MOVING COMPLETE.")
    print()
    print("Succsefuly Moved: " + str(files_moved) + " files.")
    print("Target folder originally had: " + str(target_length) + " files.")
    new_folder_size = target_length + files_moved
    print("Target folder now has: " + str(new_folder_size) + " files")
    print()



def reduce_classes():
    # 1. Rename Images in disgust by adding a letter, just so the angry and disgust images
    #    dont have duplicate names.
    rename_files(folder_path = "images_aged/train/disgust_aged", name_str = "disgust")
    rename_files(folder_path = "images_aged/val/disgust_aged", name_str = "disgust")

    # 2. Add all the images from disgust into angry.
    source_dir_train = "images_aged/train/disgust_aged"
    target_dir_train = "images_aged/train/angry_aged"
    source_dir_val = "images_aged/val/disgust_aged"
    target_dir_val = "images_aged/val/angry_aged"
    move_files(source_dir_train, target_dir_train)
    move_files(source_dir_val, target_dir_val)

    # 3. Rename angry to "angry + disgust"
    old_name_train = "images_aged/train/angry_aged"
    new_name_train = "images_aged/train/angry+disgust_aged"
    old_name_val = "images_aged/val/angry_aged"
    new_name_val = "images_aged/val/angry+disgust_aged"
    os.rename(old_name_train, new_name_train)
    os.rename(old_name_val, new_name_val)

    # 4. Delete empty disgust folder.
    os.rmdir("images_aged/train/disgust_aged")
    os.rmdir("images_aged/val/disgust_aged")

    # 5. Renaming file names for simplicity.
    new_names = ["Angry", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
    old_names = ["angry+disgust_aged", "fear_aged", "happy_aged", "neutral_aged", "sad_aged", "surprise_aged"]

    train_path = "images_aged/train"
    val_path = "images_aged/val"

    for idx ,name in enumerate(new_names):
        old_train = train_path + "/" + old_names[idx]
        new_train = train_path + "/" + name

        old_val = val_path + "/" + old_names[idx]
        new_val = val_path + "/" + name

        os.rename(old_train, new_train)
        os.rename(old_val, new_val)
    print("Renamed Folders")
    print("COMPLETED CLASS REDUCTION")


"""RUN THIS CODE TO REDUCE NUMBER OF CLASSES"""
# reduce_classes()

training_data = []
train_path = "images_aged/train"
val_path = "images_aged/val"
img_size = 224
CLASSES = ["Angry", "Fear", "Happy", "Neutral", "Sad", "Surprise"]


def create_data(train=True, img_size=img_size, train_path=train_path, val_path=val_path, shuffle=True, normalize=True, reshape=True, num_samples=10000):
    """
    Arguments:
        train: (boolean). Specifies whether we are creating a training
        or validation set.

        img_size: (int). The resize value for the dataset. By default is
        set to 224 because the pre trained models take in (224, 224) imgs.

    Output:
        return_data: list of list where each list list has a pair
        [new_array, class_label]

    Purpose:
        Get data into the correct format in order to apply transfer
        learning.

    Algorithm:
        1. Iterate over the labels.
        2. Check if training or validation set.
        3. Iterate over the images for each label.
        4. Convert the image to an array.
        5. Create list pair with array and label.
        6. Append list pair to return_data.
        7. Return return_data shuffled or not shuffled.
    """
    # This is the array that stores the data.
    return_data = []

    # Keeping track of succseful data transformations.
    succsesful_data_creations = 0

    # 1. Iterate over the labels.
    for label in CLASSES:
        print("Transforming " + label + " images into arrays.")

        # 2. Check if training or validation set.
        if train:
            path = train_path
        else:
            path = val_path


        path += "/" + label

        # 3. Iterate over the images for each label.
        class_label = label
        for img in os.listdir(path):
            try:
                # 4. Convert the image to an array.
                img_array = cv2.imread(os.path.join(path, img))

                # 5. Create list pair with array and label.
                new_array = cv2.resize(img_array, (img_size,img_size))

                # Reshaping data:
                # if reshape:
                #     new_array = new_array.reshape(img_size, img_size, 3)

                # Normalize data:
                if normalize:
                    new_array = np.array(new_array / 255.0, dtype="uint8")

                # 6. Append list pair to return_data.
                return_data.append((new_array, class_label))
                succsesful_data_creations += 1
            except:
                print("Unable to add image: " + str(img))

    print("Succsefuly created: " + str(succsesful_data_creations) + " new arrays and labels. ")
    print("CREATING DATA COMPLETE.")
    print()

    if normalize:
        print("The data was normalized.")

    # 7. Return return_data shuffled or not shuffled.
    if shuffle:
        print("The data has been shuffled.")
        print()
        random.shuffle(return_data)
        return return_data
    else:
        print("The data was not shuffled.")
        print()
        return return_data

import pickle
# print("Function Completed")

# with open('train_X.pkl', 'rb') as f:
#     X_train_normalized = pickle.load(f)
# X_train_normalized = np.array(X_train_normalized / 255.0)
# pickle.dump(X_train_normalized, open('X_train_normalized.pkl','wb'), protocol=4)
# del X_train_normalized
#
# print("Training Data Complete")
#
#
# with open('val_X.pkl', 'rb') as f:
#     X_val_normalized = pickle.load(f)
# X_val_normalized = X_val_normalized / 255.0
# pickle.dump(X_val_normalized, open('X_val_normalized.pkl','wb'), protocol=4)
# del X_val_normalized
#
# print("Validation Data Complete")


# training_data = create_data(normalize=False)
# validation_data = create_data(train=False, normalize=False)
#
#
# X = [] # Data.
# Y = [] # Labels.
# print("Made it to loop")
# for image_array, label in validation_data:
#     X.append(image_array)
#     Y.append(label)
#
# del validation_data
#
# X = np.array(X, dtype="uint8").reshape(-1, img_size, img_size, 3)
# print(X.shape)
# print(np.array(Y).shape)
#
# pickle.dump(X, open('val_X.pkl','wb'), protocol=4)
# pickle.dump(Y, open('val_Y.pkl','wb'), protocol=4)
#
# #
# #
# #
# print("Pickle complete.")
# with open('train_X.pkl', 'rb') as f:
#     X = pickle.load(f)
#
#
# X = X / 255.0
# import cv2
# arr = X[0]
# print(arr.shape)
# print(X[0])
#
# plt.imshow(cv2.cvtColor(X[0], cv2.COLOR_BGR2RGB))
# plt.show()
# print("All code executed. ")
