# Image-Classification
By: Sharan Arora, Joseph Bridges, Aditya Chawla, Sanyam Jain, Yash Jain

## Purpose

We built our website to allow users to test the performance of both of our models: a Convolutional Neural Network and a Vision Transformer. Both of these models are trained on the MNIST dataset. With our website, a user can upload a CSV file with pixel intensities of a 28x28 image meant to represent a handwritten number. The image will then be classified according to two different machine-learning models and the predicted digit will be displayed on the screen. 

We leveraged AWS Lightsail to deploy a Django instance and upload our models, trained using Google Colab. Additionally, this server-side code was linked to our client-side code using Anvil which allowed us to build a robust UI for ease of access and use.

## Approach
The MNIST dataset, a collection of handwritten digits, presents a benchmark challenge for evaluating image classification models. Our objective was to compare the effectiveness of CNNs and ViTs in accurately classifying these images, taking into consideration factors such as model complexity, training time, and accuracy.

The approach involves training both a CNN and a ViT model on the MNIST dataset and exploring different configurations for the CNN, such as varying convolutional layers, filter sizes, max pool layers, dense layers, regularizers, dropout layers, batch sizes, and number of epochs to achieve the desired accuracy. For the Vision Transformer, experimentation with different hidden, key/query, and value dimensions, the number of heads for the multi-head self-attention (MHSA) layers, and the number of layers were performed.

## Process Methodology 
Due to the computational intensity of training neural networks, a validation set approach was chosen, with TensorFlow managing the validation splits to fine-tune the models effectively.

Initially, the MNIST dataset underwent loading and normalization to prepare the images for processing. For the CNN model, additional reshaping was performed to tailor the dataset for CNN compatibility. The architecture of the CNN model was designed with layers suitable for feature extraction and classification, including convolutional, pooling, and dense layers. Similarly, the Vision Transformer (ViT) model processed the images by converting them into sequences of 2D patches and adding positional embeddings to capture spatial relationships. Both models were constructed with attention to detail, utilizing the Adam optimizer and sparse categorical cross-entropy for loss calculation, reflecting their suitability for multi-class classification tasks.

Training of both models incorporated a validation split to monitor performance and mitigate overfitting, with checkpoints implemented to preserve the iteration exhibiting optimal validation accuracy. Upon completing the training phase, the best-performing models were loaded for final evaluation on the MNIST test set, assessing their predictive accuracy. Furthermore, a detailed examination of misclassifications was conducted to identify patterns and common errors made by both models. Below are the relative accuracies, sample misclassifications, and limitations of these two models.

## Results
### CNN Model
The prediction accuracy of the CNN on the test dataset (when trained on the entire dataset) is 99.47% with a loss (according to the Sparse Categorical Cross Entropy metric) of 0.0363. The confusion matrix showed a particular difficulty with certain digits, where '3' was often misidentified as '5'. Surprisingly, the model distinguished well between '0' and '6', which can be challenging. However, it struggled to differentiate between '3' and '8', and many digits were misclassified as '8'. This suggests that the CNN model may have issues identifying the unique features that distinguish certain pairs of digits, especially when there are similarities in their closed loops or other local features.

Here, it is evident that some misclassified images present challenges for even human recognition. This highlights the fact that the inherent limitations of the data quality make achieving 100% accuracy unattainable, though a well-tuned model can approach this ideal closely.

Beyond the data's intrinsic characteristics, certain digits are prone to consistent misidentification. The model particularly has trouble telling apart numbers with loops or curves. For instance, the numbers 2 and 9 are frequently confused due to the subtle difference in the length of their minor upper bar. Similarly, distinguishing between 9 and 4 can be tricky, as it often depends on discerning whether a handwritten figure is a complete loop or includes a small extension at the top.

The misidentifications between 1 and 7, as well as 7 and 9, could stem from their visual resemblances, which are often emphasized by the way they're handwritten. In the subsequent figure, a confusion matrix is utilized to illustrate these commonly misclassified digit pairs.

### ViT Model 
The Vision Transformer Model performs comparatively worse than the CNN model, with a final accuracy of 98.53 % and a loss of 0.0739 (SCCE) in test data.

The model seems to struggle with numbers that have relatively small or long tails and/or thin strokes. The ViT model most frequently confuses 3s with 5s and 8s with 9s. Considering that these numbers can often be misinterpreted even by humans for certain handwritings, these mistakes are not so concerning. More concerning are the models’ confusion of 7s with 8s as well as 3s with 9s, which were also common mistakes. The full confusion matrix is shown below.

Listed below are the pros and cons for each:

### Pros of CNN:
CNNs are highly efficient with parameters, making them suitable for scenarios with limited data. They excel at recognizing local patterns, such as textures and shapes, due to their convolutional filters. Generally, CNNs require less computational power compared to Vision Transformers, especially for smaller datasets. Extensively tested and used in various applications, offering a wealth of research, frameworks, and pre-trained models.

### Cons of CNN:
CNNs struggle with understanding global context or relationships between distant parts of an image. Performance improvement may plateau as network depth increases, requiring more sophisticated architectures like ResNets.

### Pros of ViT:
ViTs excels at capturing global relationships within an image due to self-attention mechanisms. They can efficiently scale with the addition of more data, often outperforming CNNs in large-scale datasets. The self-attention mechanism makes it adaptable to various input sizes without changing the architecture.

### Cons of ViT:
Training ViTs from scratch requires significant computational resources and large datasets. They tend to require more data to achieve comparable performance with CNNs on smaller datasets. Also, they are more complex to understand and implement compared to traditional CNNs.

In practice, CNNs excel in scenarios where precision in local feature recognition is vital, such as in object and facial recognition tasks. ViTs stand out in environments that require a comprehensive understanding of the entire image, like in complex scene analysis or object detection in densely populated scenes. The decision to use CNNs or ViTs should be informed by the specific demands of the application, considering the type of images, dataset size, and computational constraints.

## Conclusion
Both models achieved very high accuracy, with the CNN slightly outperforming the ViT, indicating the effectiveness of deep learning for image classification tasks, even with the relatively simple architecture of the CNN compared to the more complex ViT. The confusion matrix for the ViT model shows a high rate of correct predictions, with strong true positive rates for all digits. The matrix also revealed specific patterns of misclassification, particularly among digits that share visual similarities in shape and stroke.

The visualization of misclassified instances provided practical insights into the limitations of the ViT model, identifying common misclassification pairs and highlighting the challenge of distinguishing certain digits from one another. While both models demonstrated the capacity to learn and generalize from the training data to unseen test data effectively, they still faced challenges with ambiguous and complex handwritten digits, underscoring the need for robustness in model training. The exercise suggests that while current models are highly accurate, there's still room for improvement, especially in handling edge cases and enhancing the model's ability to discriminate between visually similar categories.


