import tensorflow as tf
from tensorflow.keras import Model
import os
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)
cifar = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


class LeNet5(Model):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.c1 = tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), activation='sigmoid')
        self.p1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)
        self.c2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), activation='sigmoid')
        self.p2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)

        self.f1 = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(120, activation='sigmoid')
        self.dense2 = tf.keras.layers.Dense(84, activation="sigmoid")
        self.dense3 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        x = self.c1(inputs)
        x = self.p1(x)
        x = self.c2(x)
        x = self.p2(x)

        x = self.f1(x)
        x = self.dense(x)
        x = self.dense2(x)
        y = self.dense3(x)
        return y


model = LeNet5()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

model_savepath = 'lenet_checkpoint/lenet5.ckpt'
if os.path.exists(model_savepath + '.index'):
    print("----------------------load model-----------------------")
    model.load_weights(model_savepath)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_savepath,
                                                 save_weights_only=True,
                                                 save_best_only=True)

history = model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test),
                    validation_freq=1,
                    callbacks=[cp_callback])
model.summary()

with open("lenet5_weights.txt", 'w') as f:
    for v in model.trainable_variables:
        f.write(str(v.name) + '\n')
        f.write(str(v.shape) + '\n')
        f.write(str(v.numpy()) + '\n')

acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='training accuracy')
plt.plot(val_acc, label='validation accuracy')
plt.title("training and validation accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='training loss')
plt.plot(val_loss, label='validation loss')
plt.title("training and validation loss")
plt.legend()
plt.show()
