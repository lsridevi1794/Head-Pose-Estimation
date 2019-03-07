import cv2
import os.path
import numpy as np
import csv
import glob
import re
from six.moves import cPickle as pickle
import random
from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util

#def random_rotation(image_array: ndarray):
#    # pick a random degree of rotation between 25% on the left and 25% on the right
#    random_degree = random.uniform(-25, 25)
#    return sk.transform.rotate(image_array, random_degree)
#
#def random_noise(image_array: ndarray):
#    # add random noise to the image
#    return sk.util.random_noise(image_array,mode='gaussian')
#
#def horizontal_flip(image_array: ndarray):
##     horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
#    return image_array[:, ::-1]

def blur(image_array: ndarray):
    return cv2.GaussianBlur(image_array,(5,5),0)

def sharpen(image_array: ndarray):
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    return cv2.filter2D(image_array, -1, kernel)
#def pyramid_reduce(image_array: ndarray):
##    # add random noise to the image
#    return sk.transform.pyramid_reduce(image_array)

def data_augmentation(input_path, output_path, img_size=64, colour=True, normalisation=False):

    #Image counter
    counter = 0


    #Create the output folder if does not find it
    if not os.path.exists(output_path): os.makedirs(output_path)

    num_generated_files=0
    while num_generated_files <1:
    #Iterate through all the folder specified in the input path
        for folder in os.walk(input_path + "/"):
            print("folder",str(folder[0]))
            for image_path in glob.glob(str(folder[0]) + "/*.jpg"):
    
                #Check if there are folders which not contain the 
                #substring "Person". If there are then skip them.
                splitted = str(folder[0]).split('/')
                print("splits",splitted)
                folder_name = splitted[len(splitted)-1]
                if(("Person" in folder_name) == False): break;
                print("im path",image_path)
                #Split the image name
                image_path = image_path.replace("\\","/")
                print("modified",image_path)
                splitted = image_path.split('/')
                image_name = splitted[len(splitted)-1]
                file_name = image_name.split(".")[0]
                print("")
                print("image name",image_name)
                print("file",file_name)
                #Regular expression to split the image string
                matchObj = re.match( r'(person)(?P<id>[0-9][0-9])(?P<serie>[12])(?P<number>[0-9][0-9])(?P<tilt>[+-][0-9]?[0-9])(?P<pan>[+-][0-9]?[0-9])', file_name, re.M|re.I)
                print(matchObj)
                print("COUNTER: " + str(counter))
                print(image_path)
                print(matchObj.group(0))
                print("ID: " + matchObj.group("id"))
                print("SERIE: " + matchObj.group("serie"))
                print("NUMBER: " + matchObj.group("number"))
                print("TILT: " + matchObj.group("tilt"))
                print("PAN: " + matchObj.group("pan"))
    
                person_id = matchObj.group("id")
                person_serie = matchObj.group("serie")
                number=int(matchObj.group("number"))
                tilt = matchObj.group("tilt")
                pan = matchObj.group("pan")
    
                #Take the image information from the associated txt file        
                f=open(folder[0] +"/" + file_name + ".txt")
                lines=f.readlines()
                face_centre_x = int(lines[3])
                face_centre_y = int(lines[4])
                face_w = int(lines[5])
                face_h = int(lines[6])
                f.close
    
                #Take the largest dimension as size for the face box
                if(face_w > face_h):
                    face_h = face_w
                if(face_h > face_w):
                   face_w = face_h
                face_x = face_centre_x - (face_w/2)
                face_y = face_centre_y - (face_h/2)
    
                #Correction for aberrations
                if(face_x < 0):
                   face_x = 0
                if(face_y < 0):
                   face_y = 0
                face_x = int(face_x)
                face_y = int(face_y)
                face_h = int(face_h)
                face_w = int(face_w)
                #print("C_X: " + str(face_centre_x))
                #print("C_Y: " + str(face_centre_y))
                #print("W: " + str(face_w))
                #print("H: " + str(face_h))
                #print("X: " + str(face_x))
                #print("Y: " + str(face_y))      
    
    
                #Create the output folder if does not find it
                if not os.path.exists(output_path + "/"+'Person' + str(person_id)): os.makedirs(output_path + "/" +'Person' + str(person_id))
                
                images=image_path
                #performing random rotations on the image
                image_to_transform=cv2.imread(images)
            
                
                available_transformations = {
                'blur':blur,
                'sharpen':sharpen
#                'rotate': random_rotation,
#                'noise': random_noise,
#                'horizontal_flip': horizontal_flip
                }
                
#                num_transformations_to_apply = random.randint(1, len(available_transformations))
#                print('Number of Transformations applied:',num_transformations_to_apply)
                num_transformations_to_apply = 2
            
                num_transformations = 1
                transformed_image = None
                #counter = 93
                while num_transformations <= num_transformations_to_apply:
                    print('***********************************************************************************************************')
                    # choose a random transformation to apply for a single image
                    key = random.choice(list(available_transformations))
                    transformed_image = available_transformations[key](image_to_transform)
                    output_dir = output_path + "/" +'Person'+ str(person_id) + "/" + 'person'+str(person_id) + str(person_serie) + str(number+(93*num_transformations)) + str(tilt) + str(pan)  + ".jpg"
                    num_transformations += 1
                    #Save the image
                    
                    #Since we are applying the transformations twice, the image number has to be in sequence with the older images and the series number
                    #What can we do to loop it through the image series
                    print('output_dir = ',output_dir)
                    #insert code to save the transformed image into a folder
                    cv2.imwrite(output_dir,transformed_image)
            #        cv2.imshow(transformed_image)
                
            counter+=1

        num_generated_files+=1

def main():
    
    data_augmentation("../HeadPoseImageDatabase", "../HeadPoseImageDatabase/augmented_images", img_size=64,\
               colour=True, normalisation=False)

main()    