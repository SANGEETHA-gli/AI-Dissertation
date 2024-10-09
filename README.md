# AI-Dissertation - Hybrid Deep Learning Framework for Depression Detection Using Attention Mechanisms and Facial Expression Analysis
This project presents a Hybrid Deep Learning Framework aimed at detecting signs of depression through facial expression analysis. The system integrates attention mechanisms and leverages pretrained convolutional neural networks (CNNs) for feature extraction, combined with transformer-based attention blocks for temporal analysis. The model is trained on the AffectNet dataset, consisting of approximately 29,000 images, and is designed to classify individuals as either "Depression" or "Non-Depression" based on visual cues. Additionally, the framework is intended to extend to real-time video analysis for continuous depression monitoring.



# Overview
The goal of this research is to develop an effective deep learning-based solution for __mental health monitoring__, specifically for detecting signs of depression from facial expressions in __real-time video__. The framework uses a hybrid approach by combining the strengths of multiple architectures:

* __Feature Extraction:__ Uses __MobileNetV2__ and __ResNet50__, two state-of-the-art CNNs pretrained on large image datasets, to extract high-level spatial features from facial images.
* __Attention Mechanism:__ Incorporates an attention-based transformer block to capture temporal patterns and relevant features from the extracted spatial information, enhancing the model's ability to focus on important facial regions.
* __Binary Classification:__ The system is designed for __binary classification__, where the outputs indicate the likelihood of __"Depression"__ or __"Non-Depression"__ based on the analyzed facial expressions.
