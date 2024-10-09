# AI-Dissertation - Hybrid Deep Learning Framework for Depression Detection Using Attention Mechanisms and Facial Expression Analysis
This project presents a Hybrid Deep Learning Framework aimed at detecting signs of depression through facial expression analysis. The system integrates attention mechanisms and leverages pretrained convolutional neural networks (CNNs) for feature extraction, combined with transformer-based attention blocks for temporal analysis. The model is trained on the AffectNet dataset, consisting of approximately 29,000 images, and is designed to classify individuals as either "Depression" or "Non-Depression" based on visual cues. Additionally, the framework is intended to extend to real-time video analysis for continuous depression monitoring.



# Overview
The goal of this research is to develop an effective deep learning-based solution for __mental health monitoring__, specifically for detecting signs of depression from facial expressions in __real-time video__. The framework uses a hybrid approach by combining the strengths of multiple architectures:

* __Feature Extraction:__ Uses __MobileNetV2__ and __ResNet50__, two state-of-the-art CNNs pretrained on large image datasets, to extract high-level spatial features from facial images.
* __Attention Mechanism:__ Incorporates an attention-based transformer block to capture temporal patterns and relevant features from the extracted spatial information, enhancing the model's ability to focus on important facial regions.
* __Binary Classification:__ The system is designed for __binary classification__, where the outputs indicate the likelihood of __"Depression"__ or __"Non-Depression"__ based on the analyzed facial expressions.#


# Key Features

* __Hybrid Deep Learning Approach:__ Combines the power of CNNs for spatial feature extraction with transformer-based attention for temporal analysis.
* __Attention Mechanisms:__ Utilizes self-attention to enhance the focus on relevant facial features.
* __Facial Expression Analysis:__ Trained using the __AffectNet dataset__, focusing on identifying depression-related facial patterns from a diverse set of __29,000 images__.
* __Real-Time Video Monitoring:__ The framework can be extended to support real-time depression detection using a webcam or video feed.
* __End-to-End Training Pipeline:__ Includes data preprocessing, feature extraction, attention mechanism integration, and model evaluation.

# Usage
* __Training:__ Follow the provided scripts to preprocess the data and train the model. Adjust parameters in the configuration file as needed.
* __Testing:__ Test the trained model on new images or datasets to evaluate its depression detection accuracy.
* __Real-Time Monitoring:__ The framework can be extended for real-time video analysis using a webcam to detect signs of depression continuously.


# Results
The model was trained on approximately __29,000 images__ from the AffectNet dataset, achieving a __validation accuracy of around 95.2%__. The confusion matrix indicates robust performance in distinguishing __"Depression"__ and __"Non-Depression"__ cases, with __precision, recall, and F1-score values all exceeding 0.93__. These results demonstrate the model's potential for accurately identifying depression-related facial expressions.

# Confusion Matrix
|                                   |  __Predicted: Non-Depression__ |   __Predicted: Depression__    |
| --------------------------------- | ------------------------------ | ------------------------------ |
|__Actual: Non-Depression__         | 5005                           | 184                            | 
|__Actual: Depression__             | 462                            | 4679                           |

# Future Work
The framework can be further enhanced by:
S
* __Improving real-time performance__ for more efficient video analysis.
* __Incorporating more complex temporal modeling techniques__, such as recurrent neural networks (RNNs) or expanding the use of transformer architectures.
* __Expanding the dataset size__ for fine-tuning or transfer learning on a larger set of facial expression data to enhance generalization.

# References
* [Affectnet Dataset](https://www.kaggle.com/datasets/noamsegal/affectnet-training-data)



