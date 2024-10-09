import os
import numpy as np
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D

# Paths
PROCESSED_DIR = r'C:\Users\dsang\OneDrive\Desktop\mindwell_backend\mindwell_backend\data\processedaffectnet'
PROCESSED_SHAPE_DIR = r'C:\Users\dsang\OneDrive\Desktop\mindwell_backend\mindwell_backend\data\processed_shapes'
IMAGE_DIR = r'C:\Users\dsang\OneDrive\Desktop\mindwell_backend\mindwell_backend\data\affectnet\train'
BATCH_SIZE = 32
IMAGE_SIZE = (96, 96)  # Define image size for MobileNetV2 and ResNet50

# Ensure the shape directory exists
if not os.path.exists(PROCESSED_SHAPE_DIR):
    os.makedirs(PROCESSED_SHAPE_DIR)

# Load MobileNetV2 and ResNet50 without their top layers
mobilenetv2_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(96, 96, 3))
resnet50_model = ResNet50(weights='imagenet', include_top=False, input_shape=(96, 96, 3))

# Add global average pooling to both models to convert the feature maps to 1D vectors
mobilenetv2_output = GlobalAveragePooling2D()(mobilenetv2_model.output)
resnet50_output = GlobalAveragePooling2D()(resnet50_model.output)

# Define models for feature extraction
mobilenetv2_feature_model = Model(inputs=mobilenetv2_model.input, outputs=mobilenetv2_output)
resnet50_feature_model = Model(inputs=resnet50_model.input, outputs=resnet50_output)

# Data generator for loading images
datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    IMAGE_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',  # Binary classification (depression/non-depression)
    shuffle=False
)

# Extract MobileNetV2 and ResNet50 features
mobilenetv2_features = mobilenetv2_feature_model.predict(train_generator, steps=len(train_generator))
resnet50_features = resnet50_feature_model.predict(train_generator, steps=len(train_generator))

# Save features to .npy files
np.save(os.path.join(PROCESSED_DIR, 'mobilenetv2_train_features.npy'), mobilenetv2_features)
np.save(os.path.join(PROCESSED_DIR, 'resnet50_train_features.npy'), resnet50_features)

# Save labels
y_train = train_generator.classes
np.save(os.path.join(PROCESSED_DIR, 'y_train.npy'), y_train)

# Save the shape of the features
np.save(os.path.join(PROCESSED_SHAPE_DIR, 'mobilenetv2_shape.npy'), mobilenetv2_features.shape)
np.save(os.path.join(PROCESSED_SHAPE_DIR, 'resnet50_shape.npy'), resnet50_features.shape)

print(f"MobileNetV2 features shape: {mobilenetv2_features.shape}")
print(f"ResNet50 features shape: {resnet50_features.shape}")
print(f"Labels saved with shape: {y_train.shape}")
