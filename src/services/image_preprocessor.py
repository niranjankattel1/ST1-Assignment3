import cv2 
import numpy as np

# initial version, not finished needs to be edited - rohit

class Imagepreprocessor:
    
    def __init__(self, resized_img: tuple = (128,128)):
        self.resized_img = resized_img
        pass

    def transform(self, file_path: str):
        img = cv2.imread(str(file_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, self.resized_img)
        normalized = resize.astype("float32") / 255.0
        return normalized.flatten()

    def image_displayer(self, image):
        self.image = image
        cv2.imshow("testingwindow", image)
        cv2.waitKey(0)
