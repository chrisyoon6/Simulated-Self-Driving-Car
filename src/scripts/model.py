import tensorflow as tf
from tensorflow.keras import models
import numpy as np
from PIL import Image

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