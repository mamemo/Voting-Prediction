from tec.ic.ia.p1.models.Model import Model
import tensorflow as tf
import numpy as np


class LogisticRegression(Model):
    def __init__(self, samples_train, samples_test, prefix, regularization):
        super().__init__(samples_train, samples_test, prefix)
        self.regularization = regularization
        # HyperParameters
        self.learning_rate = 0.01
        self.training_epochs = 5000
        self.batch_size = 100
        self.display_step = 5
        self.lambd = 0.01


    def execute(self):
        dim_input = self.samples_train[0].shape[1]
        dim_output = self.samples_train[1].shape[1]
        with tf.name_scope("Declaring_placeholder"):
            X = tf.placeholder(tf.float32, [None, dim_input])
            y = tf.placeholder(tf.float32, [None, dim_output])

        with tf.name_scope("Declaring_variables"):
            W = tf.Variable(tf.zeros([dim_input, dim_output]))
            b = tf.Variable(tf.zeros([dim_output]))
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, W)

        with tf.name_scope("Declaring_functions"):
            y_techo = tf.nn.softmax(tf.add(tf.matmul(X, W), b))

        with tf.name_scope("Calculating_Loss"):
            # Original loss function
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_techo))
            #Regularization
            regularizer = None
            if self.regularization == "l1":
                regularizer = tf.contrib.layers.l1_regularizer(scale=self.lambd)
            elif self.regularization == "l2":
                regularizer = tf.contrib.layers.l2_regularizer(scale=self.lambd)
            reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            regularization_penalty = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
            regularized_loss = tf.reduce_mean(loss + regularization_penalty)

        with tf.name_scope("Declaring_Optimizer"):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(regularized_loss)

        with tf.name_scope("Training"):
            with tf.Session() as sess:
                # Initialize variables
                sess.run(tf.global_variables_initializer())
                for epoch in range(self.training_epochs):
                    loss_in_each_epoch = 0
                    _, l = sess.run([optimizer, regularized_loss], feed_dict={X: self.samples_train[0], y: self.samples_train[1]})
                    loss_in_each_epoch += l
                    # Print loss
                    if (epoch+1) % self.display_step == 0:
                        print("Epoch: {}".format(epoch + 1), "loss={}".format(loss_in_each_epoch))

                print("Optimization Finished!")

                # Test model
                correct_prediction = tf.equal(tf.argmax(y_techo, 1), tf.argmax(y, 1))
                # Calculate accuracy for test examples
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                print("Accuracy:", accuracy.eval({X: self.samples_test[0], y: self.samples_test[1]}))
