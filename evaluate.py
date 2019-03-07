import tensorflow as tf


def evaluate(X_data, y_data, output, BATCH_SIZE):
    batch_x = X_data
    batch_y = y_data

    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(batch_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()

    accuracy = sess.run(accuracy_operation, feed_dict={X: batch_x, y: batch_y})
    total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples
