import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from keras.models import load_model

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from keras.models import load_model


train_path = '/content/drive/MyDrive/DataSet/train'
test_path = '/content/drive/MyDrive/DataSet/test'
val_path = '/content/drive/MyDrive/DataSet/val'

# Define image size and batch size
img_size = (224, 224)
batch_size = 32


# Define data generators for training, validation, and testing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Split the training data into training and validation sets
)


train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

model = keras.Sequential(
    [
        # First convolutional layer
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(img_size[0], img_size[1], 3)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Second convolutional layer
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Third convolutional layer
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Flatten the output from the convolutional layers
        layers.Flatten(),
        
        # Fully connected layers
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ]
)


# Print a summary of the model architecture
model.summary()

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

epochs = 5

history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator
)

test_loss, test_acc = model.evaluate(test_generator)

print(f"Test loss: {test_loss}")
print(f"Test accuracy: {test_acc}")

model.save('/content/drive/MyDrive/SCmodel.h5')
print("Model Saved as : ScalpCareModel-Final.h5")

import json
np.save('my_history.npy',history.history)
import pandas as pd

# convert the history.history dict to a pandas DataFrame:     
hist_df = pd.DataFrame(history.history) 

# save to json:  
hist_json_file = 'history.json' 
with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)

# or save to csv: 
hist_csv_file = 'history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'],color='red',label='train')
plt.plot(history.history['val_accuracy'],color='blue',label='validation')
plt.legend()
plt.show()

plt.plot(history.history['loss'],color='red', label = 'train_loss')
plt.plot(history.history['val_loss'],color='blue', label = 'val_loss')
plt.legend()
plt.show()

class_names = {
    0: 'Alopecia Areata',
    1: 'Contact Dermatitis',
    2: 'Folliculitis',
    3: 'Head Lice',
    4: 'Lichen Planus',
    5: 'Male pattern Baldness',
    6: 'Psoriasis',
    7: 'Seborrheic Dermatitis',
    8: 'Telogen Effluvium',
    9: 'Tinea Capitis',
}

from keras import models

# Define image size and batch size
img_size = (224, 224)
batch_size = 32

# Load the pre-trained CNN model
model = models.load_model('/content/drive/MyDrive/SCmodel.h5')

# Load the test image
img = tf.io.read_file('https://images.ctfassets.net/j6utfne5ne6b/2CymBRddLBVKYWFN8bTTIz/ab36c5e05e74230fb82a9e0b880cc316/Yellow_dandruff_on_scalp.jpg?fm=webp&q=75')
img = tf.image.decode_image(img, channels=3, expand_animations=False)
interpolation = "bilinear"
interpolation = image_utils.get_interpolation(interpolation)
img = tf.image.resize(img,(224, 224), method=interpolation)
img = np.expand_dims(img, axis=0)
img = img * 1/255.0

# Make predictions on the test image
preds = model.predict(img)

#print the predicted class label
class_label = np.argmax(preds)
print('Predicted class label:', class_label)

# Get the predicted class name
class_name = class_names[class_label]
print('Predicted class name: ', class_name)
img.shape


!pip install pytest

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from keras.utils import image_utils

model = keras.models.load_model('/content/drive/MyDrive/SCmodel.h5')

img_size = (224, 224)
batch_size = 32

# Load the test image
img = tf.io.read_file('/content/drive/MyDrive/DataSet/test/Tinea Capitis/tinea_capitis_0138.jpg')
img = tf.image.decode_image(img, channels=3, expand_animations=False)
interpolation = "bilinear"
interpolation = image_utils.get_interpolation(interpolation)
img = tf.image.resize(img,(224, 224), method=interpolation)
img = np.expand_dims(img, axis=0)
img = img * 1/255.0

# Make predictions on the test image
preds = model.predict(img)

# Get the predicted class label
class_label = np.argmax(preds)
print('Predicted class label:', class_label)

# Get the predicted class name
class_names = {
    0: 'Alopecia Areata',
    1: 'Contact Dermatitis',
    2: 'Folliculitis',
    3: 'Head Lice',
    4: 'Lichen Planus',
    5: 'Male pattern Baldness',
    6: 'Psoriasis',
    7: 'Seborrheic Dermatitis',
    8: 'Telogen Effluvium',
    9: 'Tinea Capitis',
}
class_name = class_names[class_label]
print('Predicted class name: ', class_name)

# Plot the test image
import matplotlib.pyplot as plt
plt.imshow(img[0])
plt.axis('off')
plt.title('Test Image')
plt.show()
 

test_path = '/content/drive/MyDrive/DataSet/test'

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)
 
 

# Load the test image
img = tf.io.read_file('/content/drive/MyDrive/DataSet/test/Tinea Capitis/tinea_capitis_0138.jpg')
img = tf.image.decode_image(img, channels=3, expand_animations=False)
interpolation = "bilinear"
interpolation = image_utils.get_interpolation(interpolation)
img = tf.image.resize(img,(224, 224), method=interpolation)
img = np.expand_dims(img, axis=0)
img = img * 1/255.0

# Make predictions on the test image
preds = model.predict(img)

# Get the predicted class label and class name
class_label = np.argmax(preds)
class_name = class_names[class_label]

print("Predicted Class Label: ", class_label)
print("Predicted Class Name: ", class_name)

from keras import models    
 # Define image size and batch size
img_size = (224, 224)
batch_size = 32

# Load the pre-trained CNN model
model = models.load_model('/content/drive/MyDrive/SCmodel.h5')

# Compile the model
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Perform unit testing
def test_model():
    # Test the model on the test dataset
    test_loss, test_acc = model.evaluate(test_generator)
    assert test_loss is not None, "Test loss should not be None"
    assert test_acc is not None, "Test accuracy should not be None"
    assert test_loss >= 0, "Test loss should be greater than or equal to 0"
    assert test_acc >= 0, "Test accuracy should be greater than or equal to 0"

# Run the unit tests
test_model()
print("All unit tests passed.")

import numpy as np

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(15, 15))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
    
    import os
print(os.listdir())

from sklearn.metrics import confusion_matrix
import numpy as np

classes = ['Alopecia Areata', 'Contact Dermatitis', 'Folliculitis', 'Head Lice', 'Lichen Planus', 'Male Pattern Baldness', 'Psoriasis', 'Seborrheic Dermatitis', 'Telogen Effluvium', 'Tinea Capitis']

y_true = np.array(test_set.labels)
print("True : ", y_true)

y_pred = model.predict(test_set)
y_pred = np.argmax(y_pred, axis=1)
print("Predicted : ", y_pred)

conf_mat = confusion_matrix(y_true, y_pred)

plot_confusion_matrix(cm           = conf_mat,
                      normalize    = False,
                      target_names = classes,
                      title        = "ScalpCare Confusion Matrix")
