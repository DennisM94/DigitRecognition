import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# scale down/normalize training data
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()

# 1D Layer that flattens our 28x28 grid
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))

#trys to take all outputs and calculates how likely it is that the digit is what is recd
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(x_train, y_train, epochs=10)

loss, accuracy = model.evaluate(x_test, y_test)
print("Accuracy:", accuracy)
print("Loss:", loss)

model.save("digits_model")

#read the drawn digits to analye, try to 

for x in range(1, 6):
    img = cv.imread(f'DrawnDigits\\{x}.png')[:,:,0]
    img = np.invert(np.array([img]))
    
    prediction = model.predict(img)
    print(f'The result is probably: {np.argmax(prediction)}')
    
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()