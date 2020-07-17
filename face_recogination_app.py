from __future__ import division
import kivy
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.label import Label
import cv2
from matplotlib import pyplot as plt 
import numpy as np 
from math import cos, sin 

green = (0, 255, 0)

class face(App):

    def show(self, image):
        #figure size in inches
        plt.figure(figsize=(10,10))
        plt.imshow(image, interpolation='nearest')

    def overlay_mask(self, mask, image):
        #mask the mask rgb
        rgb_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        img = cv2.addWeighted(rgb_mask, 0.5, image, 0.5, 0)
        return img
    
    def find_biggest_contour(self, image):
        #copy image
        image = image.copy()
        contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        #to find biggest contour
        contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
        biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

        #return the biggest contour
        mask = np.zeros(image.shape, np.uint8)
        cv2.drawContours(mask, [biggest_contour], -1, 255, -1)
        return biggest_contour, mask
    
    def circle_contour(self, image, contour):

        #bounding ellipse
        image_with_ellipse = image.copy()
        ellipse = cv2.fitEllipse(contour)
        #add it
        cv2.ellipse(image_with_ellipse, ellipse, green, 2, cv2.LINE_AA)
        return image_with_ellipse

    def find_object(self, image):
        #RGB is red green blue
        #BGR is blue green red
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        #to scale the image
        max_dimension = max(image.shape)
        scale = 700/max_dimension
        image = cv2.resize(image, None, fx=scale, fy=scale)

        #to clean the image 
        image_blur = cv2.GaussianBlur(image, (7,7), 0)
        image_blur_hsv = cv2.cvtColor(image_blur, cv2.COLOR_RGB2HSV)

        #defining the filters by color
        min_red = np.array([0,100, 80])
        max_red = np.array([10,256, 256])

        mask1 = cv2.inRange(image_blur_hsv, min_red, max_red)

        #filter by brightness
        min_red2 = np.array([170,100,80])
        max_red2 = np.array([100,256,256])

        mask2 = cv2.inRange(image_blur_hsv, min_red2, max_red2)

        mask = mask1 + mask2

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)

        big_object_contour, mask_object = find_biggest_contour(mask_clean)

        overlay = overlay_mask(mask_clean, image)

        circled = circle_contour(overlay, big_object_contour)
        show(circled)

        #convert back to original color scheme
        bgr = cv2.cvtColor(circled, cv2.COLOR_RGB2BGR)
        return bgr
        
        image = cv2.imread('apple.jpeg')
        result = self.find_object(image)
        #write the new image
        cv2.imwrite('apple2.jpeg', result)
    
    #to read the image
    def call(self):
        image = cv2.imread('apple.jpeg')
        result = self.find_object(image)
        #write the new image
        cv2.imwrite('apple2.jpeg', result)

if __name__ == '__main__':
    face().run()
    #img = Image(source = "apple2.jpeg")
    #print(img)
    cv2.destroyAllWindows()