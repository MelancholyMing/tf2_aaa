import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

model_save_path = "./checkpoint/mnist.ckpt"
if os.path.exists(model_save_path + ".index"):
    print("------------load model---------------")
    model.load_weights(model_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)
history = model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=1,
                    callbacks=[cp_callback])

model.summary()
print(model.trainable_variables)

with open("./weight.txt", 'w') as f:
    for v in model.trainable_variables:
        f.write(str(v.name) + "\n")
        f.write(str(v.shape) + '\n')
        f.write(str(v.numpy()) + '\n')
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='training accuracy')
plt.plot(val_acc, label='validation accuracy')
plt.title('training and validation accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='training loss')
plt.plot(val_loss, label='validation loss')
plt.title('training and validation loss')
plt.legend()
plt.show()
