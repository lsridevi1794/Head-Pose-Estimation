import tensorflow as tf

def trainer(network, Y):

    learning_rate = 1e-6
    #find error like squared error but bette0r
    # cross_entropy=tf.nn.softmax_cross_entropy_with_logits_v2(logits=
    # network,labels=Y)

    cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=Y, logits=network)
    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=network)
    #now minize the above error
    #calculate the total mean of all the errors from all the nodes
    cost=tf.reduce_mean(cross_entropy)

    #Now backpropagate to minimise the cost in the network.
    optimizer=tf.train.AdamOptimizer(learning_rate).minimize(cost)
    return [optimizer, cost]
