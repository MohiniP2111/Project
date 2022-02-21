import tensorflow as tf
import numpy as np
import cv2
import time

class_names = ['a', 'd', 'f', 'h', 'n', 'sa', 'su']
expression_codewords = {'a': 'anger',
                        'd': 'disgust',
                        'f': 'fear',
                        'h': 'happiness',
                        'n': 'neutral',
                        'sa': 'sadness',
                        'su': 'surprise',
                        }
img_height = 180
img_width = 180
input_image_path = "12137399835_d9075d3194_b.jpg"
test_image = cv2.imread(input_image_path)
test_image_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
haar_cascade_face = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')
faces_rects = haar_cascade_face.detectMultiScale(test_image_gray, scaleFactor=1.2, minNeighbors=5)
for (x, y, w, h) in faces_rects:
    cv2.rectangle(test_image, (x, y), (x + w, y + h),
                  (0, 0, 255), 2)

    faces = test_image[y:y + h, x:x + w]
    cv2.imwrite('temp_1.jpeg', faces)
time.sleep(0.50)
path = r'temp_1.jpeg'

# Reading an image in default mode
image = cv2.imread(path)

# Window name in which image is displayed
window_name = 'Input Image'

# Using cv2.imshow() method
# Displaying the image
cv2.imshow(window_name, image)

# waits for user to press any key
# (this is necessary to avoid Python kernel form crashing)
cv2.waitKey(3)
cv2.destroyWindow(window_name)
img = tf.keras.utils.load_img(
    'temp_1.jpeg', target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
# cv2.imshow('hh',img_array)
# cv2.waitKey()
img_array = tf.expand_dims(img_array, 0)  # Create a batch
model = tf.keras.models.load_model('model_1.h5')
predictions = model.predict(img_array)
print(predictions)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(expression_codewords[class_names[np.argmax(score)]], 100 * np.max(score))
)
cv2.imshow(expression_codewords[class_names[np.argmax(score)]], image)
cv2.waitKey(0)
