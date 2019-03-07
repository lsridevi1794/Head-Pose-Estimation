import input_fn, model, training
import numpy as np
import math
import os
import tensorflow as tf
import model
import evaluate

# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


ROOT_PATH = "../HeadPoseImageDatabase"
train_data_directory = os.path.join(ROOT_PATH, "training")
batch_size = 256

label_csv_path = train_data_directory+"/prima_label.csv"
filenames = list(np.genfromtxt(label_csv_path, delimiter=',', skip_header=1, usecols=(0), dtype=None))
labels= list(np.genfromtxt(label_csv_path, delimiter=',', skip_header=1, usecols=5, dtype=None))
# X = tf.placeholder(tf.float32, shape=[None,64,64,3])
# Y = tf.placeholder(tf.float32, shape=[None,25])

# X = inputs['images']
# Y = inputs['labels']

X = tf.placeholder(name= 'ip', dtype=tf.float32, shape=(None, 64,64,1))
Y = tf.placeholder(tf.int32, [None,1])
# network = model.model(X_train)
network = model.cnn_model(X)
[optimizer,cost] = training.trainer(network, Y)
print(optimizer)

# Initialization
sess = tf.Session()
num_points = len(filenames)
# Run training
for epoch in range(100):
    for jj in range(int(math.floor((num_points // batch_size) - 1))):
        # Get the data
        sess.run(tf.global_variables_initializer())
        inputs = input_fn.input_fn(filenames, labels, batch_size)
        sess.run(inputs['iterator_init_op'])
        train_X = sess.run(inputs['images'])
        train_Y = sess.run(inputs['labels'])
        train_Y = np.array(train_Y)
        train_Y = train_Y.reshape((batch_size,1))

        # input_to_sess = {X:train_X, Y:train_Y}
        temp = sess.run(X, feed_dict={X:train_X})
        sess.run(optimizer,feed_dict = {X:train_X, Y:train_Y})
        # print('Done with batch {} and epoch {}'.format(jj,epoch))

    # Evaluate the model
    train_X = sess.run(inputs['images'])
    train_Y = sess.run(inputs['labels'])
    train_Y = np.array(train_Y)
    train_Y = train_Y.reshape((batch_size,1))

    output = sess.run(network, feed_dict = {X:train_X})
    pred= tf.argmax(output,1)
    corr_pred = tf.equal(pred, train_Y)
    accuracy_operation = tf.reduce_mean(tf.cast(corr_pred,
    tf.float32))
    for class_label in range(3):
        corr_pred_class= tf.equal(pred==class_label, train_Y==class_label)
        class_accuracy= tf.reduce_mean(tf.cast(corr_pred_class, tf.float32))
        print("Accuracy after epoch {} for class {} is {}".format(epoch, class_label, sess.run(class_accuracy)))
    print('Accuracy after epoch {} is {}'.format(epoch,
    sess.run(accuracy_operation))) 

    


# session.run(tf.global_variables_initializer())
#     counter=0
#     for epoch in range(epochs):
#         tools = processing_tools()
#         for batch in range(int(number_of_images / batch_size)):
#             counter+=1
#             images, labels = tools.batch_dispatch()
#             if images == None:break
#             loss = session.run([cost], feed_dict=
#                            {images_ph: images, labels_ph: labels})
#             print('loss', loss)
#             session.run(optimizer, feed_dict={images_ph: images,
#                                               labels_ph: labels})
#             print('Epoch number ', epoch, 'batch', batch,
#                                                          'complete')
