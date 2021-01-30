import tensorflow as tf

W = tf.Variable(tf.constant(5, dtype=tf.float32))
lr = 0.01
epoch = 40

for epoch in range(epoch):
    with tf.GradientTape() as tape:
        loss = tf.square(W + 1)
    grads = tape.gradient(loss, W)
    W.assign_sub(lr * grads)
    print("EPOCH:", epoch, "W:", W.numpy(), "loss:%f"%loss)
    # print("After %s epoch,w is %f,loss is %f" % (epoch, W.numpy(), loss))
