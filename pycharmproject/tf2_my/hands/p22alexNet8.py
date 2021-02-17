import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, MaxPool2D, Dropout, Flatten, Dense

cifar = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


class AlexNet8(Model):
    def __init__(self):
        super(AlexNet8, self).__init__()
        self.c1 = Conv2D(filters=96, kernel_size=(3, 3))
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        self.p1 = MaxPool2D(pool_size=(3, 3), strides=2)

        self.c2 = Conv2D(filters=256, kernel_size=(3, 3))
        self.b2 = BatchNormalization()
        self.a2 = Activation('relu')
        self.p2 = MaxPool2D(pool_size=(3, 3), strides=2)

        self.c3 = Conv2D(filters=384, kernel_size=(3, 3), padding="same", activation='relu')
        self.c4 = Conv2D(filters=384, kernel_size=(3, 3), padding="same", activation='relu')
        self.c5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')
        self.p3 = MaxPool2D(pool_size=(3, 3), strides=2)

        self.flatten = Flatten()
        self.f1 = Dense(2048, activation='relu')
        self.d1 = Dropout(0.5)
        self.f2 = Dense(2048, activation='relu')
        self.d2 = Dropout(0.5)
        self.f3 = Dense(10, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)

        x = self.c2(x)
        x = self.b2(x)
        x = self.a2(x)
        x = self.p2(x)

        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)
        x = self.p3(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.d1(x)
        x = self.f2(x)
        x = self.d2(x)
        y = self.f3(x)
        return y


model = AlexNet8()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

model_savepath = "./alexnet8_checkpoint/alexnet8.ckpt"
if os.path.exists(model_savepath + '.index'):
    print("-------------------load model--------------------")
    model.load_weights(model_savepath)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_savepath,
                                                 save_weights_only=True,
                                                 save_best_only=True)

history = model.fit(x_train, y_train, batch_size=32,
                    epochs=50, validation_data=(x_test, y_test),
                    validation_freq=1, callbacks=[cp_callback])

model.summary()

with open('alexnet_weights.txt', 'w') as f:
    for v in model.trainable_variables:
        f.write(str(v.name) + '\n')
        f.write(str(v.shape) + '\n')
        f.write(str(v.numpy()) + '\n')

acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='training acc')
plt.plot(val_acc, label='validation acc')
plt.title('training and validation acc')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='training loss')
plt.plot(val_loss, label='validation loss')
plt.title('training and validation loss')
plt.legend()
plt.show()
