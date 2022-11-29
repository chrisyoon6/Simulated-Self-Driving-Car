import os
from PIL import Image as Image_PIL
import cv2
import numpy as np

class Helpers:
    """This class contains any useful helper functions.
    """    
    def __init__(self) -> None:
        pass

    @staticmethod
    def move_frames(folder, new_folder, start_frames_count):
        """Moves all frames from a folder to another folder. Renames the frames count such that the labeled files can be unique.

        Args:
            folder (str): path of the directory to move files from
            new_folder (str): path of the directory to move files to
            start_frames_count (int): 
        Raises:
            ValueError: If the input starting frames count is present in the new folder.
        """        
        count = start_frames_count
        file_list = os.listdir(folder)
        file_list.sort()
        hsv, frameNum, *vels = file_list[-1].split('_')
        if (frameNum >= start_frames_count):
            raise ValueError(f"Input start frame count is not unique - Largest={frameNum}, input={start_frames_count}")

        for filename in file_list:
            img = np.array(Image_PIL.open(os.path.join(folder, filename)))
            hsv,frameNum,x,z = filename.split('_')
            new_filename = "_".join([hsv, str(count), str(x), str(z)])
            cv2.imwrite(os.path.join(new_folder, new_filename), img)
            count += 1
    
def main():
    pass

if __name__ == "__main__":
    main()