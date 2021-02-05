import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(len(x_train), 28, 28, 1)

image_gen_train = ImageDataGenerator(
    rescale=1. / 255.,
    rotation_range=45,
    width_shift_range=.15,
    height_shift_range=.15,
    horizontal_flip=False,
    zoom_range=0.5
)
image_gen_train.fit(x_train)
print("x_train: ", x_train.shape)
x_train_sub1 = np.squeeze(x_train[:12])
print("x_train_sub1: ", x_train_sub1.shape)
x_train_sub2 = x_train[:12]
print("x_train_sub2: ", x_train_sub2.shape)

fig = plt.figure(figsize=(20, 2))
plt.set_cmap('gray')

for i in range(0, len(x_train_sub1)):
    ax = fig.add_subplot(1, 12, i + 1)
    ax.imshow(x_train_sub1[i])
fig.suptitle("subset of original training images", fontsize=20)
plt.show()

fig = plt.figure(figsize=(20, 2))
for x_batch in image_gen_train.flow(x_train_sub2, batch_size=32, shuffle=False):
    print(x_batch)
    for i in range(0, 12):
        ax = fig.add_subplot(1, 12, i + 1)
        ax.imshow(np.squeeze(x_batch[i]))
    fig.suptitle('augmented images', fontsize=20)
    plt.show()
    break
