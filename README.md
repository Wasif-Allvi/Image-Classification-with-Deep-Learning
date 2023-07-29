
# Image Classification using Deep Learning


### Project Overview:

This repository presents an Image Classification project that aims to differentiate between honey bees and bumble bees using Convolutional Neural Networks (CNNs). The primary objective is to build a robust model capable of accurately identifying these bee species, despite the challenges posed by variations in backgrounds, positions, and image resolutions.


## Data Description

This repository contains a dataset sourced from DataCamp, including various data files and image data. The dataset consists of labeled images, with each image belonging to one of two classes: honey bee (0.0) or bumble bee (1.0). The labels are stored in a CSV file, where the index corresponds to the image name (e.g., "1036.jpg"), and the "genus" column indicates the bee type (0.0 for honey bee, 1.0 for bumble bee).


![](https://github.com/Wasif-Allvi/Image-Classification-with-Deep-Learning/assets/45194832/4160e4ba-3ab7-4dcc-a512-081e87fff091)



![bombus](https://github.com/Wasif-Allvi/Image-Classification-with-Deep-Learning/assets/45194832/41581c53-c21e-4d1e-bf9e-2a560b38e7c4)




## Dependencies

The project relies on the following libraries: pickle, pathlib, skimage, pandas, numpy, scikit-learn, keras, Dense, Dropout, Flatten, Conv2D, MaxPooling2D.
## Data Splitting and Normalization

- The image data matrix, X, and the corresponding labels, y, are split into train, test, and evaluation sets.
- 20% of the data is allocated to the evaluation (holdout) set, which is exclusively used to assess the model's performance after training.
- The remaining data is split 60/40 into train and test sets, which are utilized during the supervised learning process.
- To ensure consistent learning across all features (color channels), data normalization is performed. The StandardScaler from scikit-learn scales the data, setting the mean to 0 and the standard deviation to 1 for each color channel individually.

Normalization is a crucial preprocessing step that involves scaling the pixel values of our images to have a standard range. The goal is to bring each feature, which corresponds to a color channel in our RGB images, to a similar scale. By doing so, we prevent certain features from dominating the learning process, allowing the neural network to learn from all features equally.

The need for normalization arises due to the varying pixel value ranges present in images. In RGB images, pixel values can range from 0 to 255 for each color channel. Without normalization, features with larger ranges can overshadow others, leading to biased learning and potential information loss. By normalizing the data, we transform the pixel values to have a mean of 0 and a standard deviation of 1. This standardization process enables the neural network to focus on learning patterns and relationships across all color channels, enhancing its ability to extract valuable insights from the data.

To achieve normalization, we leverage sklearn's StandardScaler, which automates the math involved in scaling the data. For each color channel independently, the StandardScaler subtracts the mean of the pixel values across the entire dataset from each pixel value and then divides it by the standard deviation. This individual treatment ensures that each color channel is normalized appropriately, avoiding any bias introduced by channel-specific differences.
## Model Architecture

The project utilizes Convolutional Neural Networks (CNNs) to tackle the image classification task. The model architecture is as follows:

- Sequential Model: A linear stack of layers.
- Two Convolutional Layers: These layers employ kernels to extract features from the images.
- MaxPooling: This passes a (2, 2) moving window over the image and downscales the image by outputting the maximum value within the window.
- Conv2D: This adds a third convolutional layer since deeper models, i.e. models with more convolutional layers, are better able to learn features from images.
- Dropout: This prevents the model from overfitting, i.e. perfectly remembering each image, by randomly setting 25% of the input units to 0 at each update during training.
- Flatten: As its name suggests, this flattens the output from the convolutional part of the CNN into a one-dimensional feature vector which can be passed into the following fully connected layers.
- Dense: Fully connected layer where every input is connected to every output (see image below).
- Dropout: Another dropout layer to safeguard against overfitting, this time with a rate of 50%.
- Dense: Final layer which calculates the probability the image is either a bumble bee or honey bee.

Note that: The total Trainable parameters are : 3,669,249. There is no Non-trainable parameters.



## Model Training and Evaluation

- The model is compiled for training, specifying the loss function, optimizer, and metric.
- A 'mock' training is initially performed using a small subset of data for a limited number of epochs to get a sense of the process.
- The final training involves multiple iterations over the entire dataset.
- The model's performance is assessed on both the test set and the evaluation set.
- Evaluation metrics include test loss, test accuracy, evaluation loss, and evaluation accuracy.


We get the Test loss: 0.671, Test accuracy: 0.618 and Evaluation loss: 0.670, Evaluation accuracy: 0.637. This means,

- The model's performance, as indicated by test accuracy and evaluation accuracy, is around 61.8% and 63.7%, respectively.
- The evaluation accuracy is slightly higher than the test accuracy, suggesting that the model is generalizing reasonably well to unseen data.
- The loss values for both test and evaluation sets are relatively close, indicating that the model's predictions align with the true labels with acceptable consistency.
- The provided accuracy values show the model's ability to distinguish between honey bees and bumble bees, but there is room for improvement.

## Model Predictions

The model is used to predict probabilities and class predictions for individual images using the .predict() method and np.round().

Showing first 5 probabilities and first five class predictions below:

[[0.5104395 ]
 [0.5508864 ]
 [0.6333447 ]
 [0.18111488]
 [0.33957684]]  and [[1]
 [1]
 [1]
 [0]
 [0]]

### Probabilities

- The probabilities represent the model's confidence scores for each image being classified as either a bumble bee or a honey bee.
- A probability value close to 1 indicates that the model is highly confident that the image belongs to the predicted class and vice versa.
- For example, in the first image, the model is approximately 51.04% confident that it is a bumble bee (class 1) and 48.96% confident that it is a honey bee (class 0).

### Class Predictions

- The class predictions are derived from the probabilities by rounding each probability value to the nearest integer (0 or 1).
- A class prediction of 1 indicates that the model predicts the image to be a bumble bee and vice versa.
- For example, in the first image, the model predicts a class of 1 (bumble bee) because the probability (0.5104395) is closer to 1 than 0.


## Epoch

The improvement in accuracy over time and the corresponding decrease in loss are noticed, and eventually, the accuracy levels off. The occurrence of plots like these can aid in diagnosing overfitting. If an upward curve in the validation loss had been observed as time went on (a U shape in the plot), it would have led to the suspicion that the test set was starting to be memorized by the model, and its generalization to new data might have been compromised.

As evident from the graph, the model demonstrates a favorable trend where the loss consistently decreases as the number of epochs increases. Simultaneously, the accuracy steadily improves during the training process. This positive behavior indicates that the model is effectively learning from the training data and generalizing well to new, unseen data. The absence of an upward curve in the validation loss (a U shape) further confirms that the model is not overfitting or memorizing the test data. Instead, it shows promising potential for accurate predictions on new, real-world data, showcasing its ability to generalize effectively.




![](https://github.com/Wasif-Allvi/Image-Classification-with-Deep-Learning/assets/45194832/89f99b09-f41c-4d0f-b96f-e20a4b92dca8)
## Improvement


Given the current performance, potential improvements may involve adjusting the model's architecture, fine-tuning hyperparameters, increasing the size of the training dataset, or exploring transfer learning to leverage pre-trained models on larger image datasets for better accuracy.

Your valuable contributions are highly encouraged in this project. Whether you have innovative ideas for enhancements or you spot any issues that need fixing, don't hesitate to open an issue or submit a pull request.

I trust that this project serves as a source of inspiration, igniting your curiosity to explore the boundless potential of Deep Learning and predictive modeling. Happy Coding!
