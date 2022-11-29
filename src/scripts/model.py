import tensorflow as tf
from tensorflow.keras import models
import numpy as np
import cv2
from PIL import Image

print("--------------------Model script ----------------------------")
class Model:
    """This class is responsble for handling trained models.

    NOTES:
    - when loading images, use PIL to open image. Opening with cv2 expands the dimensions to 3, even
    if the image has been saved as 2D. Resaving an image that has been opened with cv2 will expand the image dimensions, even if it
    is loaded again with PIL
    """
    def __init__(self, path) -> None:
        """Creates a Model object, representing a trained cnn that can be used.

        Args:
            path (str): path where the trained model is saved. 
        """        
        self.mod = models.load_model(path)
        print(type(self.mod))
    
    @staticmethod
    def preprcocess_img(img):
        """Processes the input image to a format that can be compared with the cnn.

        Args:
            img (cv::Mat): input image

        Returns:
            cv::Mat: processed image
        """        
        img = img / 255
        img = np.expand_dims(np.expand_dims(img,axis=-1),axis=0)
        return img

    def predict(self, img):
        """Predicts what the robot's velocities should be based on the input image.
        Args:
            img (cv::Mat): input image, without normalization or dimension expansion, just in the same format as the scraped images.

        Returns:
            np.array: A 1-D array containing the model's predictions
        """        
        img = Model.preprcocess_img(img)
        pred = self.mod.predict(img)[0]
        return pred

def temp():
    path = '/home/fizzer/ros_ws/src/models/drive_model_comp.h5'
    md = Model(path)
    # input = np.zeros((1,720,1280,1))
    input = np.array(Image.open('/home/fizzer/ros_ws/src/ENPH353-Team12/src/drive-data-hsv-compressed/hsv_1_0.5_0.png'))
    input = input/255
    input = np.expand_dims(np.expand_dims(input,axis=-1),axis=0)
    print(input.shape)
    print(md.predict(input))

def temp2():
    path = '/home/fizzer/ros_ws/src/models/drive_model_comp.h5'
    mod = Model(path)
    input = np.array(Image.open('/home/fizzer/ros_ws/src/ENPH353-Team12/src/drive-data-hsv-compressed/hsv_1_0.5_0.png'))
    pred = mod.predict(input)
    print(pred)

def main():
    # temp()
    temp2()
    
if __name__ == '__main__':
    main()