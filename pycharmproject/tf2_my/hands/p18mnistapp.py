import tensorflow as tf
import numpy as np
from PIL import Image

model_save_path = './checkpoint/mnist.ckpt'

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.load_weights(model_save_path)

preNum = int(input('the number of test picture:'))

for i in range(preNum):
    img_path = input("the path of picture:")
    img = Image.open(img_path)
    img = img.resize((28, 28), Image.ANTIALIAS)
    img_arr = np.array(img.convert("L"))


    # img_arr = img_arr - 255.0
    for i in range(28):

        for j in range(28):
            if img_arr[i][j] < 200:
                img_arr[i][j] = 255
            else:
                img_arr[i][j] = 0
    img_arr = img_arr / 255.0
    print("img_arr:", img_arr.shape)
    x_predict = img_arr[tf.newaxis, ...]
    print("x_predict:", x_predict.shape)
    result = model.predict(x_predict)
    pred = tf.argmax(result, axis=1)
    print("\n")
    tf.print(pred)
