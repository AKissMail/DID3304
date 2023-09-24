import tensorflow as tf
"""
This is a simple implementation of an classification for the mnist dataset. The Neural network reaches around 97% of 
accuracy. In the first block we normalize the dataset (value between 0 and 1) and we have here test and training data.
Here we download the Mnist dataset from the Tensorflow project and do some pre-processing. 
"""
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
"""
Here we define the the neural network with:
    - input layer of 784 neurons for the B/W picture 
    - hidden layer of 256 neurons
    - output layer of 10 neurons
    
We use to different activation function
    - relu returns 0 for negative inputs anf the input for positive inputs
    - softmax returns a 1 over all neurons in the layer. It shows us a likelihood for each case
"""
MyModel = tf.keras.Sequential()
MyModel.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# hidden layer with 128 neurons and relu as an activation function
MyModel.add(tf.keras.layers.Dense(256, activation='relu'))
# output layer with 10 neurons and softmax as an activation function
MyModel.add(tf.keras.layers.Dense(10, activation='softmax'))
"""
Here we run and train the model with 3 iterations of the training data. 
"""
MyModel.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
MyModel.fit(x_train, y_train, epochs9)
MyModel.save("Handwritten.model")
"""
Test the model and print out the results of the test data. 
"""
loss, accuracy = MyModel.evaluate(x_test, y_test)
print("The Model has a accuracy of" + str(accuracy))