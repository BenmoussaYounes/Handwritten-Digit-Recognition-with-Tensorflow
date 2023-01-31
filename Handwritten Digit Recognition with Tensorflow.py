import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3)
loss, accuarcy = model.evaluate(x_test, y_test)

print('ACCUARCY', accuarcy)
print('LOSS', loss)
model.save('digits.model')


for x in range(1, 10):
   img = cv.imread(f'digits.model/assets/{x}.png')[:,:,0]
   img = np.invert(np.array([img]))
   prediction = model.predict(img)
   print(f'The result is probably : {np.argmax(prediction)}')
   plt.imshow(img[0], cmap=plt.cm.binary)
   plt.show()
   #cv.imshow('image', img)




