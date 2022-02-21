import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from cf_matrix import make_confusion_matrix

sys_details = tf.sysconfig.get_build_info()
cuda_version = sys_details["cuda_version"]
print('CUDA')
print(cuda_version)
cudnn_version = sys_details["cudnn_version"]
print('CUDNN')
print(cudnn_version)
# Defininn some parameters for the loader
batch_size = 32
img_height = 180
img_width = 180
# data_dir = os.path.join(os.getcwd(), "DATASET")
data_dir = "D:\\FER_DESERTATION\\CODE\\DATASET"
# It's good practice to use a validation split when developing your model.
# Let's use 60% of the images for training, and 20% for validation. 20 for testing
## 80%
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.4,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
## 20%
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
## 20%

# test_ds = val_ds[int(len(val_ds)/2):]
# val_ds = val_ds[:int(len(val_ds)/2)]
test_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=45654654,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names
# Get names of each class
print(class_names)
# Plotting images in a pic just to visualize them together
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()

for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    print(labels_batch)
    break
# print('#####s')
# print(test_ds[1])
'''
Let's make sure to use buffered prefetching so you can yield data from disk without having I/O become blocking. 
These are two important methods you should use when loading data:

Dataset.cache keeps the images in memory after they're loaded off disk during the first epoch. 
This will ensure the dataset does not become a bottleneck while training your model. 
If your dataset is too large to fit into memory, you can also use this method to create a performant on-disk cache.
Dataset.prefetch overlaps data preprocessing and model execution while training.
Interested readers can learn more about both methods, as well as how to cache data to disk in the Prefetching section of the Better performance with the tf.data API guide.
'''

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = layers.Rescaling(1. / 255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

num_classes = 7
'''
The Sequential model consists of three convolution blocks (tf.keras.layers.Conv2D) with a max pooling layer (tf.keras.layers.MaxPooling2D) in each of them. There's a fully-connected layer (tf.keras.layers.Dense) with 128 units on top of it that is activated by a ReLU activation function ('relu'). This model has not been tuned for high accuracy—the goal of this tutorial is to show a standard approach'''

model = Sequential([
    layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(2, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(2, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    # layers.Conv2D(16, 3, padding='same', activation='relu'),
    # layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(6, activation='relu'),
    layers.Dense(num_classes)
    ##########################################
    ##  7 values addition 1 [[v1 , v2,v3 ... v7]]
    #########################################
])

'''
For this tutorial, choose the tf.keras.optimizers.Adam optimizer and tf.keras.losses.SparseCategoricalCrossentropy loss function. To view training and validation accuracy for each training epoch, pass the metrics argument to Model.compile.
'''

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
'''
Printing the model summary
'''
print(model.summary())

epochs = 15
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)
###
model.save('CNN_Model_long.h5')

###
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
plt.savefig('Tran_VAl_loss.png')

####


######
labelss = []
predictionss = []
for image_batch, labels_batch in test_ds:
    y_pred = model.predict(image_batch)
    labelss.extend(labels_batch.numpy())
    y = []
    for x in y_pred:
        score = tf.nn.softmax(x)
        y.append(np.argmax(score))
        predictionss.append(np.argmax(score))

con_mat = tf.math.confusion_matrix(labels=labelss, predictions=predictionss).numpy()
print(classification_report(labelss, predictionss,
                            target_names=['anger', 'disgust', 'fear', 'happy', 'neutral', 'suprise', 'sad']))
make_confusion_matrix(con_mat, class_names, class_names,
                      title=f"CNN confusion matrix on Test Dataset after {epochs} epochs")
plt.show()
# true_categories = tf.concat([y for x, y in test_ds], axis=0)
#
# disp = confusion_matrix(predicted_categories, true_categories)
# plt.show()
####

# sunflower_path = "12137399835_d9075d3194_b.jpg"
# img = tf.keras.utils.load_img(
#     sunflower_path, target_size=(img_height, img_width)
# )
# img_array = tf.keras.utils.img_to_array(img)
# img_array = tf.expand_dims(img_array, 0)  # Create a batch
#
# predictions = model.predict(img_array)
# print(predictions)
# score = tf.nn.softmax(predictions[0])
#
# print(
#     "This image most likely belongs to {} with a {:.2f} percent confidence."
#     .format(class_names[np.argmax(score)], 100 * np.max(score))
# )