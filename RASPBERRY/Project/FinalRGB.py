import sys
import cv2
import tflite_runtime.interpreter as tflite
import numpy as np
import time
import RPi.GPIO as GPIO
from hx711 import HX711
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.segmentation import active_contour
 
 
image = 0

index = ['Pomelo Sweetie', 'Cherry Wax Red', 'Tamarillo', 'Beetroot', 'Mandarine', 'Grape Blue',
         'Apple Golden 1', 'Grape White 4', 'Kiwi', 'Mangostan', 'Rambutan', 'Corn Husk',
         'Lemon Meyer', 'Pear Williams', 'Blueberry', 'Cherry Rainier', 'Tomato 4', 'Pear Kaiser',
         'Cactus fruit', 'Pepper Orange', 'Guava', 'Kaki', 'Potato White', 'Apple Red Yellow 2',
         'Physalis with Husk', 'Watermelon', 'Walnut', 'Salak', 'Tomato 1', 'Plum 2', 'Hazelnut',
         'Pepino', 'Apple Red 1', 'Grapefruit Pink', 'Peach', 'Apple Pink Lady', 'Apple Red 3',
         'Pineapple', 'Peach 2', 'Pear', 'Pepper Yellow', 'Onion White', 'Pear Red', 'Corn',
         'Pear Abate', 'Strawberry Wedge', 'Banana', 'Cherry Wax Yellow', 'Apple Golden 3',
         'Tomato Heart', 'Strawberry', 'Apple Red Yellow 1', 'Apple Red Delicious',
         'Granadilla', 'Cocos', 'Avocado', 'Tomato not Ripened', 'Pear Monster',
         'Peach Flat', 'Physalis', 'Banana Lady Finger', 'Mango Red', 'Limes', 'Dates',
         'Potato Sweet', 'Tangelo', 'Cherry 1', 'Potato Red Washed', 'Pear 2',
         'Apple Granny Smith', 'Tomato 3', 'Huckleberry', 'Pepper Green', 'Plum',
         'Grape White', 'Mulberry', 'Nut Pecan', 'Clementine', 'Carambula', 'Potato Red',
         'Melon Piel de Sapo', 'Nectarine Flat', 'Cantaloupe 1', 'Grapefruit White',
         'Tomato Cherry Red', 'Cherry 2', 'Nectarine', 'Onion Red', 'Grape White 2',
         'Apple Crimson Snow', 'Papaya', 'Maracuja', 'Pitahaya Red', 'Cherry Wax Black',
         'Tomato Maroon', 'Pomegranate', 'Apple Red 2', 'Redcurrant', 'Kohlrabi', 'Pear Stone',
         'Grape White 3', 'Cantaloupe 2', 'Cauliflower', 'Tomato 2', 'Quince', 'Apricot', 'Fig',
         'Passion Fruit', 'Eggplant', 'Ginger Root', 'Tomato Yellow', 'Lemon', 'Nut Forest', 'Pepper Red',
         'Plum 3', 'Apple Golden 2', 'Lychee', 'Kumquats', 'Mango', 'Apple Braeburn', 'Avocado ripe',
         'Cucumber Ripe', 'Onion Red Peeled', 'Pear Forelle', 'Pineapple Mini', 'Raspberry', 'Chestnut',
         'Grape Pink', 'Banana Red', 'Orange', 'Cucumber Ripe 2']


def getClass(output_data):
    biggest = 0
    j = 0
    k = 0
    for i in output_data[0]:
        if i > biggest:
            j = k
            biggest = i
        k = k + 1
            
    return [index[j], str(biggest)]
        
# Segmentation constants
# This defines the area to be segmented
s = np.linspace(0, 2*np.pi, 400)
r = 155 + 200*np.sin(s)
c = 190 + 200*np.cos(s)
init = np.array([r, c]).T
def segmentImage():
    global image
    
    img = rgb2gray(image)
    snake = active_contour(gaussian(img, 3, preserve_range=False),
                           init, alpha=0.015, beta=10, gamma=0.001)
    
    originalH, originalW = img.shape
    xMax = 0
    yMax = 0
    yMin, xMin = img.shape
    for coordx in snake[:, 1]:
        if coordx > xMax:
            xMax = int(coordx)
        if coordx < xMin:
            xMin = int(coordx)
            
    for coordy in snake[:, 0]:
        if coordy > yMax:
            yMax = int(coordy)
        if coordy < yMin:
            yMin = int(coordy)
            
    #Making the cut squared to not loose aspect ratio
    #width = xMax - xMin
    #height = yMax - yMin
    
   # diff = abs(width - height)
    #diff = int(diff / 2)
    #if (width > height):
    #    yMin = yMin - diff
    ##    yMax = yMax + diff
    #    if (yMin < 0):
    #        yMin = 0
   #     if (yMax > originalH):
    #        yMax = originalH
    
    #if (height > width):
    #    xMin = xMin - diff
    #    xMax = xMax + diff
    #    if xMin < 0:
    #        xMin = 0
    #    if xMax > originalW:
    #        xMax = originalH
            
    return [xMin, yMin, xMax, yMax]
    

#Camera constants
cam = cv2.VideoCapture(0)
def takePhoto():
    global image
    while True:
        ret, image = cam.read()
        if ret == True:
            # image.shape = (310, 380, 3)
            image = image[90:400, 155:535] 
            break 


# TFLite constants
# Load the TFLite model and allocate tensors.
interpreter = tflite.Interpreter(model_path="/home/pi/Desktop/Project/modelRGB.tflite")
interpreter.allocate_tensors()
# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
def makeInterpretation():
    global image
    # preparing for the inference
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, dsize=(100, 100))
    img = np.array(img, dtype=np.float32)
    img = img / 256
    img = np.expand_dims(img, 0)
    
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

def savePhoto(name):
    global image
    cv2.imwrite(name, image)

# Main logic of the program after we have the weight from the scale
def mainLogic(weight):

    global image
    
    # Take photo
    print("Taking photo...")
    takePhoto()
    
    # Saving the original photo
    print("Saving the original photo...")
    savePhoto('/home/pi/Desktop/Project/original.jpg')
    
    # Getting segmentation bounding
    print("segmenting...")
    bound = segmentImage()
    print(bound)
    
    # Cutting the image to the bounding segment
    print("Cutting the image...")
    image = image[bound[1]:bound[3], bound[0]:bound[2]]
    
    # Saving the segmented photo
    print("Saving the segmented photo...")
    savePhoto('/home/pi/Desktop/Project/segmented.jpg')
    
    # Classifing
    print("Doing interpretation...")
    classes = makeInterpretation()
    
    # Output the result
    print("Getting class...")
    finalClass = getClass(classes)
    print(finalClass)
    
    # Printing the result
    print("You have a: " + finalClass[0] + ", with a weight of: " + str(weight) + ", with a certanty of: " + finalClass[1])
    return
    
    
# Scale constants and config
referenceUnit = -1875
hx = HX711(5, 6)
def readScale():
    global image
    
    hx.set_reading_format("MSB", "MSB")
    hx.set_reference_unit(referenceUnit)
    hx.reset()
    hx.tare()
    
    isObjectInScale = False
    
    
    while True:
        try:
            val = hx.get_weight(5)
            #print(val)
            hx.power_down()
            hx.power_up()
            
            if val > 10.0 and isObjectInScale == False:
                print("Weighting...")
                isObjectInScale = True
                time.sleep(3) # Waiting time to stabilize the reading
                val = hx.get_weight(5)
                hx.power_down()
                hx.power_up
                mainLogic(val)
                
            if val <= 10.0:
                print("Scale ready!")
                print("Put fruit on the scale")
                isObjectInScale = False

        except (KeyboardInterrupt, SystemExit):
            cleanAndExit()


readScale()
