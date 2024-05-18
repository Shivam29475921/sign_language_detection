import os
import numpy as np
from PIL import Image


# custom class to dynamically create our dataset from folders
class GetData:

    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.x_train = []
        self.y_train = []

    def load_data(self):
        def convert_image_to_array(image_path):
            image = Image.open(image_path)
            # converting each image into an array(x_train)
            image_array = np.array(image)
            # appending their subsequent folder name(y_train)
            self.y_train.append(image_path[8])
            return image_array

        def convert_folder_to_array(folder):
            image_files = os.listdir(folder)
            image_arrays = []

            for each_image in image_files:
                image_array = convert_image_to_array(os.path.join(folder, each_image))
                image_arrays.append(image_array)
            return image_arrays

        for each_folder in self.folder_path:
            self.x_train.extend(convert_folder_to_array(each_folder))

        # converting our datasets to np.array before returning them
        self.x_train = np.array(self.x_train)
        self.y_train = np.array(self.y_train)
        return self.x_train, self.y_train



