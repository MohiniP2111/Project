import sys
import os
import shutil
import cv2
from multiprocessing import Process
import time

def convertToRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Extracted VisualImageData Directory
data_dir = "\VisualImageData\\"
# Counter to maintain total images count
image_counter = 0
# Get number of CPU's in host system and use -2 to allow other programs to run smoothly
num_cpus = os.cpu_count() - 2
# Empty list to store files
files_list = []
# Total data counter
total_data = 0

#  'a', 'd', 'f', 'h', 'n', 'sa' and 'su' represent
# respectively 'anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness' and 'surprise'
expression_codewords = {'a': 'anger',
                        'd': 'disgust',
                        'f': 'fear',
                        'h': 'happiness',
                        'n': 'neutral',
                        'sa': 'sadness',
                        'su': 'surprise',
                        }
# Creating directories for each emotion
[os.makedirs("DATASET\\"+x, exist_ok=True) for x in expression_codewords.keys()]
# Creating a face detector object 
haar_cascade_face = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')

# Function to detect faces and store them
def sort_files(files: list,temppp):
    """

    :param files: A list of lists where files[i] = [input_file_path, output_file_path]
    :param temppp: Temporary parameter
    """
    print(f'THREAD STARTED with {len(files)}')
    # For each read, write in files
    for read, write in files:
        # Read the input file
        test_image = cv2.imread(read)
        # Converting to gray scale
        test_image_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        # Running detector on the gray scale image
        faces_rects = haar_cascade_face.detectMultiScale(test_image_gray, scaleFactor=1.2, minNeighbors=5)
        # Position of the face
        for (x, y, w, h) in faces_rects:
            # Drawing a rectangle
            cv2.rectangle(test_image, (x, y), (x + w, y + h),
                          (0, 0, 255), 2)
            # Cropping the face
            faces = test_image[y:y + h, x:x + w]
            # Saving it to the write path
            cv2.imwrite(write, faces)


if __name__ == '__main__':
    # Recursively traversing the directory and finding all images
    for root, dirs, files in os.walk(data_dir, topdown=False):
        for name in files:
            # get the name of the folder example a01, su1 etc
            folder_name = os.path.basename(os.path.normpath(root))
            # Append input and output_file_path to the master list
            # if the first to charecters of the folder name are charecters then take them else only use the first charecter
            # The reason behind this is a is for anger , and su is for suprise
            files_list.append([os.path.join(root, name),
                               os.path.join(os.getcwd(), "DATASET",
                                            folder_name[0:2] if folder_name[0:2].isalpha() else folder_name[0:1],
                                            str(image_counter) + ".jpeg")
                               ])
            # Update the image counter
            image_counter += 1
    # A list to keep the parallel processes in memory
    process_list = []
    # Another counter to keep track parallel processes
    coutterr = 0
    # Per cpu files store the batch size of parallel processes
    per_cpu_files = image_counter // num_cpus
    print(f"[INFO] Processing {per_cpu_files} on each CPU.")
    # For each CPU
    for cpu in range(num_cpus):
        # getting indexes to slice data
        next_range = coutterr + per_cpu_files
        # Creating Process object for each slice of data
        p1 = Process(target=sort_files, args=(files_list[coutterr:next_range], coutterr))
        # Start Process
        p1.start()
        # Append Process and store to list
        process_list.append(p1)
        # update the range for slicing
        coutterr += coutterr + per_cpu_files
        # time.sleep(2)
