# Importing the TensorFlow/Keras libraries and packages
import os
import sys
import numpy as np
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

# Initialising the CNN
classifier = Sequential()
classifier.add(Input(shape=(64, 64, 3)))

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), activation='relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

# Compiling the CNN
classifier.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Part 2 - Fitting the CNN to the images
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    'dataset/training_set',
    target_size=(64, 64),
    batch_size=1,
    class_mode='binary'
)

test_set = test_datagen.flow_from_directory(
    'dataset/test_set',
    target_size=(64, 64),
    batch_size=1,
    class_mode='binary'
)

classifier.fit(
    training_set,
    epochs=25,
    validation_data=test_set
)

# Part 3 - Making new predictions
# Use o caminho da imagem como argumento opcional:
# python cnn.py dataset/single_prediction/pizza/12718.jpg
DEFAULT_IMAGE = os.path.join('dataset', 'single_prediction', 'pizza', '12718.jpg')

if len(sys.argv) > 1:
    image_path = sys.argv[1]
else:
    if os.path.exists(DEFAULT_IMAGE):
        image_path = DEFAULT_IMAGE
    else:
        image_path = None
        for label in ['pizza', 'not_pizza']:
            folder = os.path.join('dataset', 'single_prediction', label)
            if os.path.isdir(folder):
                files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if files:
                    image_path = os.path.join(folder, files[0])
                    break
        if image_path is None:
            raise FileNotFoundError(
                'Nenhuma imagem de predição encontrada em dataset/single_prediction/pizza ou not_pizza. '
                'Coloque uma imagem em dataset/single_prediction/pizza/ ou dataset/single_prediction/not_pizza/ '
                'ou passe o caminho como argumento.'
            )

print(f'Usando imagem para predição: {image_path}')

test_image = image.load_img(image_path, target_size=(64, 64))

test_image = image.img_to_array(test_image)
test_image = test_image / 255.0
test_image = np.expand_dims(test_image, axis=0)

result = classifier.predict(test_image, verbose=0)

print(result[0][0])

# Melhor prática: usar limiar ao invés de comparação direta
if result[0][0] > 0.5:
    prediction = 'pizza'
else:
    prediction = 'not_pizza'

print(prediction)