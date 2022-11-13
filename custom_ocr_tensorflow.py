import tensorflow
tensorflow.__version__

import numpy as np
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint

(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

train_data.shape, test_data.shape

train_labels.shape, test_labels.shape

digits_data = np.vstack([train_data, test_data])
digits_labels = np.hstack([train_labels,test_labels])

digits_data.shape
digits_labels.shape

index = np.random.randint(0,digits_data.shape[0])
plt.imshow(digits_data[index], cmap="gray")
plt.title('Class: '+ str(digits_labels[index]))

sns.countplot(digits_labels)

# !wget https://iaexpert.academy/arquivos/alfabeto_A-Z.zip

zip_object = zipfile.ZipFile(file='/content/alfabeto_A-Z.zip', mode='r')
zip_object.extractall('./')
zip_object.close()

dataset_az = pd.read_csv('/content/A_Z Handwritten Data.csv').astype('float32')
dataset_az

alphabet_data = dataset_az.drop('0', axis = 1)
alphabet_labels = dataset_az['0']

alphabet_data.shape

alphabet_labels

alphabet_data = np.reshape(alphabet_data.values, (alphabet_data.shape[0],28,28))

index = np.random.randint(0,alphabet_data.shape[0])
plt.imshow(alphabet_data[index],cmap='gray')
plt.title('Class: '+ str(alphabet_labels[index]))

sns.countplot(alphabet_labels)

digits_labels, np.unique(digits_labels)

alphabet_labels, np.unique(alphabet_labels)

alphabet_labels += 10

np.unique(alphabet_labels)

data = np.vstack([alphabet_data,digits_data])
labels = np.hstack([alphabet_labels,digits_labels])

print(data.shape, labels.shape)

print(np.unique(labels))

data = np.array(data, dtype='float32')

data = np.expand_dims(data, axis = -1)

print(data[0].min(), data[0].max())

# Normalization of the datais necessary in order to get faster results since there is a huge diff between the min and max values

data /= 255.0

print(data.shape)

print(np.unique(labels), len(np.unique(labels)))

le = LabelBinarizer()
labels = le.fit_transform(labels)

print(np.unique(labels))

plt.imshow(data[0].reshape(28,28), cmap="gray")
plt.title("Class: "+ str(labels[0]))

classes_total = labels.sum(axis=0)
print(classes_total)



classes_weight = {}
for i in range(0,len(classes_total)):
    classes_weight[i] = classes_total.max() / classes_total[i]

print(classes_weight)

X_train,X_test,y_train,y_test = train_test_split(data, labels, test_size=0.2, random_state = 1, stratify=labels)



augmentation = ImageDataGenerator(rotation_range = 10, zoom_range = 0.05, width_shift_range = 0.1, height_shift_range = 0.1, horizontal_flip= False)

network = Sequential()

network.add(Conv2D(filters = 32, kernel_size= (3,3), activation = 'relu', input_shape = (28,28,1)))
network.add(MaxPool2D(pool_size=(2,2)))

network.add(Conv2D(filters = 64, kernel_size= (3,3), activation = 'relu', padding= "same"))
network.add(MaxPool2D(pool_size=(2,2)))

network.add(Conv2D(filters = 128, kernel_size= (3,3), activation = 'relu', padding = "valid"))
network.add(MaxPool2D(pool_size=(2,2)))

network.add(Flatten())

network.add(Dense(64, activation="relu"))
network.add(Dense(128, activation="relu"))
network.add(Dense(36, activation="softmax"))

network.compile(loss="categorical_crossentropy", optimizer = "adam", metrics = ['accuracy'])

network.summary()

name_labels = "0123456789"
name_labels += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
name_labels = [l for l in name_labels]

# Training nueral network

file_model = "custom_ocr.model"
epochs = 30
batch_size = 128

check_pointer = ModelCheckpoint(file_model, monitor="val_loss", verbose=1, save_best_only=True)

history = network.fit(augmentation.flow(X_train,y_train,batch_size=batch_size), validation_data = (X_test, y_test),
steps_per_epoch = len(X_train) // batch_size,
epochs = epochs, class_weight = classes_weight,
verbose =1, callbacks = [check_pointer])

predictions = network.predict(X_test, batch_size=batch_size)

print(predictions[0])

print(len(predictions[0]))

print(np.argmax(predictions[0]))

print(name_labels[np.argmax(predictions[0])])

print(y_test[0])

network.evaluate(X_test,y_test)

print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1),target_names=name_labels))


plt.plot(history.history['val_loss'])

plt.plot(history.history['val_accuracy'])
