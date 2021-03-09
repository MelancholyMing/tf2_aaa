import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

input_word = 'abcdefghijklmnopqrstuvwxyz'
w_to_id = {i: j for j, i in enumerate(input_word)}

data_set_scaled = [i for i in range(len(input_word))]
x_train, y_train = [], []
for i in range(4, 26):
    x_train.append(data_set_scaled[i - 4:i])
    y_train.append(data_set_scaled[i])

# state = np.random.get_state()
# np.random.set_state(state)
np.random.seed(7)
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)
tf.random.set_seed(7)

x_train = np.reshape(x_train, (len(x_train), 4))
y_train = np.array(y_train)

model = tf.keras.models.Sequential([
    Embedding(26, 2),
    SimpleRNN(10),
    Dense(26, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])
model_save_path = './rnn_embedding_4pre1_checkpoint/rnn_embedding_4pre1.ckpt'
if os.path.exists(model_save_path + '.index'):
    print('====================load model==========================')
    model.load_weights(model_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 monitor='loss')

history = model.fit(x_train, y_train, batch_size=32, epochs=100, callbacks=[cp_callback])
model.summary()

with open("./rnn_embedding_4pre1_weights.txt", 'w') as f:
    for v in model.trainable_variables:
        f.write(str(v.name) + '\n')
        f.write(str(v.shape) + '\n')
        f.write(str(v.numpy()) + '\n')

acc = history.history['sparse_categorical_accuracy']
loss = history.history['loss']

plt.plot(acc, label='training accuracy')
plt.plot(loss, label='training loss')
plt.title("training accuracy and loss")
plt.legend()
plt.show()

preNum = int(input("input number of test alphabet:"))
for i in range(preNum):
    alphabet1 = input("input testing alphabet:")
    alphabet = [w_to_id[j] for j in alphabet1]
    alphabet = np.reshape(alphabet, (1, 4))

    result = model.predict(alphabet)
    pred = int(tf.argmax(result, axis=1))

    tf.print(alphabet1 + '->' + input_word[pred])
