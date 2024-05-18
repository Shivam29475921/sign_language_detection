from DatasetCreation import GetData
import keras
from sklearn.utils import shuffle

# dataset
FOLDERS = ["../Data/A/", "../Data/B/", "../Data/C/"]


# manual shuffling
def shuffle_data(count, data1, data2):
    for i in range(count):
        return shuffle(data1, data2)


# this class will return a custom model for sign language prediction
class TrainingModel:
    def __init__(self):
        self.model = None
    
    def create_model(self, folder_path):
        # getting data for our x_train, y_train from GetData class
        obj = GetData(folder_path)
        x_train, y_train = obj.load_data()
        y_train = [ord(x) - 65 for x in y_train]  # converting y_train from an array of chars to an array of ints
        x_train = x_train / 255
        '''
            converting y_train into a one hot encoded dataset
            so if y = [2, 1, 3] and y can have values in [1, 4]
            y_categorical = [[0,1,0,0],[1,0,0,0],[0,0,1,0]]
            this allows us to use categorical_crossentropy as our loss function 
            instead of sparse_categorical_crossentropy 
        '''
        y_train_categorical = keras.utils.to_categorical(
            y_train, num_classes=len(folder_path)
        )
        # performing manual shuffling
        x_train, y_train_categorical = shuffle_data(10, x_train, y_train_categorical)
        
        # model creation
        self.model = keras.Sequential([
            keras.layers.Flatten(input_shape=(175, 175)),
            keras.layers.Dense(1000, activation='relu'),
            keras.layers.Dense(500, activation='relu'),
            keras.layers.Dense(len(folder_path), activation='sigmoid')
        ])
        # model compilation(using categorical_crossentropy)
        self.model.compile(
            optimizer='SGD',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        #  model fit using our collected dataset
        self.model.fit(x_train, y_train_categorical, epochs=50, shuffle=True)
        self.model.save('model.keras')
        return self.model


customAnn = TrainingModel()  # keras neural network
customModel = customAnn.create_model(FOLDERS)
