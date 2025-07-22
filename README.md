# Brain-Tumor-Detection-App
Data Set Description
The image data that was used for this problem is Brain MRI Images for Brain Tumor Detection. It conists of MRI scans of two classes:
NO - no tumor, encoded as 0
YES - tumor, encoded as 1

Data Processing
First, we create a data list for storing image data in numpy array form
Secondly, we create a paths list for storing paths of all images
Thirdly, we create result list for storing one hot encoded form of target class whether normal or tumor
The label 0 is transformed into [1, 0] (one-hot encoding).
The label 1 is transformed into [0, 1] (one-hot encoding).

Model Description
In this step, we are constructing an AI model that uses a Convolutional Neural Network (CNN), which is particularly good for image recognition tasks. Our model begins with two convolutional layers that have 32 filters each; these layers are designed to detect basic patterns in the brain MRI images, like edges and textures.

Batch normalization
We apply batch normalization after the convolutional layers to accelerate training by scaling the outputs to a standard range.

Pooling
Next, we introduce a pooling layer to reduce the dimensionality of the data, which helps the model to focus on the important features, and a dropout layer to prevent overfitting, which is when the model learns the training data too well and performs poorly on new data. We repeat this pattern of convolutional, batch normalization, pooling, and dropout layers with 64 filters in the convolutional layers to capture more complex patterns.

Fully connected layer
After processing through these layers, the data is flattened into a one-dimensional array so it can be fed into densely connected layers, which will make the final decisions about what the patterns represent – in our case, whether there is a tumor or not.

Activation function and optimizer
The last dense layer uses softmax activation to output probabilities for each class, which completes our model architecture. We compile the model with a categorical crossentropy loss function, which is suitable for multi-class classification problems, and choose the Adamax optimizer, an adaptation of the Adam optimizer that is designed to work well with models that have embeddings and sparse data.

CNN Architecture:

<img width="1040" height="320" alt="Typical_cnn" src="https://github.com/user-attachments/assets/9dab5241-38ca-4bd1-9468-c97096af0c85" />

Schematic diagram of a Convolutional Neural Network (CNN): starting with an input image, the network applies multiple convolutional layers to detect features, interspersed with subsampling (pooling) layers to reduce dimensionality, and culminates in fully connected layers that lead to the final output classification

Source: Wikipedia
