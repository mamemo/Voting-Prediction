from tec.ic.ac.p1.models.Model import Model
import tensorflow as tf
import numpy as np


class LogisticRegression(Model):
    def __init__(self, samples, prefix, regularization, dim_input, dim_output):
        super().__init__(samples, prefix)
        self.dim_output = dim_output
        self.dim_input = dim_input
        self.regularization = regularization
        # Parameters
        self.learning_rate = 0.01
        self.training_epochs = 25
        self. batch_size = 100
        self.display_step = 1


    def execute(self):
        # tf Graph Input
        X = tf.placeholder(tf.float32, [self. batch_size, self.dim_input])  # Inputs of the model
        Y = tf.placeholder(tf.float32, [self. batch_size, self.dim_output])  # Outputs of the model

        # Set model weights
        W = tf.Variable(tf.random_normal([self.dim_input, self.dim_output], stddev=0.1))
        b = tf.Variable(tf.zeros([self.dim_output]))

        logits = tf.matmul(X, w) + b
        entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)


        # Construct model
        #pred = tf.nn.softmax(tf.matmul(x, W) + b)  # Softmax

        # Minimize error using cross entropy
        #cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))

        loss = tf.reduce(entropy)  # computes the mean over examples in the batch

        # Gradient Descent
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cost)

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()

        # Start training
        with tf.Session() as sess:

            # Run the initializer
            sess.run(init)

            # Training cycle
            for epoch in range(self.training_epochs):
                avg_cost = 0.
                total_batch = int(mnist.train.num_examples / self.batch_size)
                # Loop over all batches
                for i in range(total_batch):
                    batch_xs, batch_ys = mnist.train.next_batch(self.batch_size)
                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                                  y: batch_ys})
                    # Compute average loss
                    avg_cost += c / total_batch
                # Display logs per epoch step
                if (epoch + 1) % self.display_step == 0:
                    print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

            print("Optimization Finished!")

            # Test model
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
