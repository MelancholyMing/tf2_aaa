import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import os
from PIL import Image

np.set_printoptions(threshold=np.inf)
# fashion = tf.keras.datasets.fashion_mnist
# (x_train, y_train), (x_test, y_test) = fashion.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0

train_path = "./FASHION_FC/fashion_image_label/fashion_train_jpg_60000/"
train_txt = "./FASHION_FC/fashion_image_label/fashion_train_jpg_60000.txt"
x_train_savepath = "./FASHION_FC/fashion_image_label/fashion_x_train.npy"
y_train_savepath = "./FASHION_FC/fashion_image_label/fashion_y_train.npy"

test_path = "./FASHION_FC/fashion_image_label/fashion_test_jpg_10000/"
test_txt = "./FASHION_FC/fashion_image_label/fashion_test_jpg_10000.txt"
x_test_savepath = "./FASHION_FC/fashion_image_label/fashion_x_test.npy"
y_test_savepath = "./FASHION_FC/fashion_image_label/fashion_y_test.npy"


def generateds(data_path, txt):
    x, y_ = [], []
    with open(txt, 'r') as f:
        for line in f.readlines():
            value = line.split()
            img_path = os.path.join(data_path, value[0])
            img = Image.open(img_path)
            img = np.array(img.convert("L"))
            img = img / 255.0
            x.append(img)
            y_.append(value[1])
            print("loading...:", line)

    x = np.array(x)
    y_ = np.array(y_).astype(np.int64)
    return x, y_


if not (os.path.exists(x_train_savepath) and os.path.exists(y_train_savepath) and os.path.exists(
        x_test_savepath) and os.path.exists(y_test_savepath)):
    print("----------------generate data-------------------")
    x_train, y_train = generateds(train_path, train_txt)
    x_test, y_test = generateds(test_path, test_txt)

    print("---------------save data---------------------")
    x_train_save = np.reshape(x_train, (len(x_train), -1))
    x_test_save = np.reshape(x_test, (len(x_test), -1))

    np.save(x_train_savepath, x_train_save)
    np.save(y_train_savepath, y_train)
    np.save(x_test_savepath, x_test_save)
    np.save(y_test_savepath, y_test)

print("--------------load data---------------------")
x_train_save = np.load(x_train_savepath)
y_train = np.load(y_train_savepath)
x_test_save = np.load(x_test_savepath)
y_test = np.load(y_test_savepath)
x_train = np.reshape(x_train_save, (len(x_train_save), 28, 28))
x_test = np.reshape(x_test_save, (len(x_test_save), 28, 28))

# 数据增强
# x_train = x_train.reshape(len(x_train), 28, 28, 1)
#
# image_gen_train = ImageDataGenerator(
#     rescale=1. / 1.,
#     rotation_range=45,
#     width_shift_range=.15,
#     height_shift_range=.15,
#     horizontal_flip=True,
#     zoom_range=0.5
# )
#
# image_gen_train.fit(x_train)

# fig = plt.figure(figsize=(20, 2))
# plt.set_cmap('gray')
#
# for i in range(12):
#     ax = fig.add_subplot(1, 12, i + 1)
#     ax.imshow(np.squeeze(x_train[i]))
# fig.suptitle("Original Training Images", fontsize=20)
# plt.show()
#
# fig = plt.figure(figsize=(20, 2))
# for x in image_gen_train.flow(x_train[:12], batch_size=12, shuffle=False):
#     for i in range(12):
#         ax = fig.add_subplot(1, 12, i + 1) 
#         ax.imshow(np.squeeze(x[i]))
#     fig.suptitle("augmented Images", fontsize=20)
#     plt.show()
#     break

# -----------------------------------------------------------------------
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

model_save_path = "./fashion_checkpoint/fashion.ckpt"
if os.path.exists(model_save_path + ".index"):
    print("--------------load model------------------")
    model.load_weights(model_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)
# history = model.fit(image_gen_train.flow(x_train, y_train, batch_size=32), epochs=10, validation_data=(x_test, y_test),
#                     validation_freq=1,
#                     callbacks=[cp_callback])
history = model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test),
                    validation_freq=1,
                    callbacks=[cp_callback])
model.summary()

with open("./fashion_weights.txt", 'w') as f:
    for v in model.trainable_variables:
        f.write(str(v.name) + '\n')
        f.write(str(v.shape) + '\n')
        f.write(str(v.numpy()) + '\n')

acc = history.history['sparse_categorical_accuracy']
val_acc = history.history["val_sparse_categorical_accuracy"]
loss = history.history['loss']
val_loss = history.history["val_loss"]

plt.subplot(1, 2, 1)
plt.plot(acc, label='training accuracy')
plt.plot(val_acc, label='validation_accuracy')
plt.title('training and validation accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='training loss')
plt.plot(val_loss, label='validation loss')
plt.title("training and validation loss")
plt.legend()
plt.show()
