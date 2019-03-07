import cv2
import os.path
import numpy as np
import csv
import glob
import re
import mpmath


from six.moves import cPickle as pickle

def create_csv(input_path, output_path, img_size=64, colour=True, normalisation=False):

    #Image counter
    counter = 0
    roll = 0.0

    #Create the output folder if does not find it
    if not os.path.exists(output_path): os.makedirs(output_path)

    #Write the header
    fd = open(output_path + '/prima_label.csv','w')
    fd.write("path, id, serie, tilt, pan, tilt_class, pan_class" + "\n")
    fd.close()

    #Iterate through all the folder specified in the input path
    for folder in os.walk(input_path + "/"):
        for image_path in glob.glob(str(folder[0]) + "/*.jpg"):

            #Check if there are folders which not contain the
            #substring "Person". If there are then skip them.
            splitted = str(folder[0]).split('/')
            folder_name = splitted[len(splitted)-1]
            if(("Person" in folder_name) == False): break;

            #Split the image name
            splitted = image_path.split('/')
            image_name = splitted[len(splitted)-1]
            file_name = image_name.split(".")[0]
            #Regular expression to split the image string
            matchObj = re.match( r'(person)(?P<id>[0-9][0-9])(?P<serie>[12])(?P<number>[0-9][0-9])(?P<tilt>[+-][0-9]?[0-9])(?P<pan>[+-][0-9]?[0-9])', file_name, re.M|re.I)

            person_id = matchObj.group("id")
            person_serie = matchObj.group("serie")
            tilt = int(matchObj.group("tilt"))
            pan = int(matchObj.group("pan"))

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
            face_x = int(face_centre_x - (face_w/2))
            face_y = int(face_centre_y - (face_h/2))

            #Correction for aberrations
            if(face_x < 0):
               face_x = 0
            if(face_y < 0):
               face_y = 0


            #Load the image (colour or grayscale)
            if(colour==True): image = cv2.imread(image_path) #load in colour
            else: image = cv2.imread(image_path, 0) #load in grayscale
                #Crop the face from the image
            image_cropped = np.copy(image[face_y:face_y+face_h, face_x:face_x+face_w])
            #Rescale the image to the predifined size
            image_rescaled = cv2.resize(image_cropped, (img_size,img_size), interpolation = cv2.INTER_AREA)
            #Create the output folder if does not find it
            if not os.path.exists(output_path + "/" + str(person_id)): os.makedirs(output_path + "/" + str(person_id))
            #Save the image
            output_dir = output_path + "/" + str(person_id) + "/" + str(int(person_id)) + "_" + str(tilt) + "_" + str(pan) + "_" + str(counter) + ".jpg"
            cv2.imwrite(output_dir, image_rescaled)

            #Write the CSV file for pan
            if( (pan >= -90) and (pan <= -50)) : label_pan = "FR"
            elif((pan >= -51) and (pan <= -10)): label_pan = "SR"
            elif((pan >= -11) and (pan <= 10)): label_pan = "C"
            elif((pan >= 11) and (pan <= 50)): label_pan = "SL"
            elif((pan >= 51) and (pan <= 90)): label_pan = "FL"
            else: raise ValueError('ERROR: The pan is out of range ... ' + str(pan))

            #Write the CSV file for tilt
            if((tilt >= -90) and (tilt <= -31)): label_tilt = "FD"
            elif((tilt >= -30) and (tilt <= -11)): label_tilt = "SD"
            elif((tilt >= -10) and (tilt <= 10)): label_tilt = "C"
            elif((tilt >= 11) and (tilt <= 30)): label_tilt = "SU"
            elif((tilt >= 31) and (tilt <= 90)): label_tilt = "FU"
            else: raise ValueError('ERROR: The tilt is out of range ... ' + str(tilt))

            #Write the CSV file
            fd = open(output_path + '/prima_label.csv','a')
            # if str(label_pan+label_tilt) == "FLFU" : class_label = 1
            # elif str(label_pan+label_tilt) == "FLSU": class_label = 2
            # elif str(label_pan+label_tilt) == "FLC": class_label = 3
            # elif str(label_pan+label_tilt) == "FLSD": class_label = 4
            # elif str(label_pan+label_tilt) == "FLFD": class_label = 5
            # elif str(label_pan+label_tilt) == "SLFU" : class_label = 6
            # elif str(label_pan+label_tilt) == "SLSU": class_label = 7
            # elif str(label_pan+label_tilt) == "SLC": class_label = 8
            # elif str(label_pan+label_tilt) == "SLSD": class_label = 9
            # elif str(label_pan+label_tilt) == "SLFD": class_label = 10
            # elif str(label_pan+label_tilt) == "CFU" : class_label = 11
            # elif str(label_pan+label_tilt) == "CSU": class_label = 12
            # elif str(label_pan+label_tilt) == "CC": class_label = 13
            # elif str(label_pan+label_tilt) == "CSD": class_label = 14
            # elif str(label_pan+label_tilt) == "CFD": class_label = 15
            # elif str(label_pan+label_tilt) == "SRFU" : class_label = 16
            # elif str(label_pan+label_tilt) == "SRSU": class_label = 17
            # elif str(label_pan+label_tilt) == "SRC": class_label = 18
            # elif str(label_pan+label_tilt) == "SRSD": class_label = 19
            # elif str(label_pan+label_tilt) == "SRFD": class_label = 20
            # elif str(label_pan+label_tilt) == "FRFU" : class_label = 21
            # elif str(label_pan+label_tilt) == "FRSU": class_label = 22
            # elif str(label_pan+label_tilt) == "FRC": class_label = 23
            # elif str(label_pan+label_tilt) == "FRSD": class_label = 24
            # elif str(label_pan+label_tilt) == "FRFD": class_label = 25
            if str(label_pan) == "FL" or str(label_pan) == "SL": pan_class = 0
            elif str(label_pan) == "C": pan_class = 1
            elif str(label_pan) == "FR" or str(label_pan) == "SR": pan_class = 2

            if str(label_tilt) == "FU" or str(label_tilt) == "SU": tilt_class = 0
            elif str(label_tilt) == "C": tilt_class = 1
            elif str(label_tilt) == "FD" or str(label_tilt) == "SD": tilt_class = 2


            fd.write(output_dir + "," + str(int(person_id)) + "," + str(int(person_serie)) + "," + \
                     str(label_tilt) + "," + str(label_pan) + "," + str(tilt_class) + "," + str(pan_class)+"\n")
            fd.close()
            counter += 1

if __name__ == '__main__':
    create_csv("../HeadPoseImageDatabase", "../HeadPoseImageDatabase/training", img_size=64,\
               colour=True, normalisation=False)
