import os
import numpy as np
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D

PROCESSED_DIR = r'C:\Users\dsang\OneDrive\Desktop\mindwell_backend\mindwell_backend\data\processedaffectnet'
PROCESSED_SHAPE_DIR = r'C:\Users\dsang\OneDrive\Desktop\mindwell_backend\mindwell_backend\data\processed_shapes'
IMAGE_DIR = r'C:\Users\dsang\OneDrive\Desktop\mindwell_backend\mindwell_backend\data\affectnet\train'
BATCH_SIZE = 32
IMAGE_SIZE = (96, 96)  


if not os.path.exists(PROCESSED_SHAPE_DIR):
    os.makedirs(PROCESSED_SHAPE_DIR)


mobilenetv2_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(96, 96, 3))
resnet50_model = ResNet50(weights='imagenet', include_top=False, input_shape=(96, 96, 3))


mobilenetv2_output = GlobalAveragePooling2D()(mobilenetv2_model.output)
resnet50_output = GlobalAveragePooling2D()(resnet50_model.output)


mobilenetv2_feature_model = Model(inputs=mobilenetv2_model.input, outputs=mobilenetv2_output)
resnet50_feature_model = Model(inputs=resnet50_model.input, outputs=resnet50_output)

datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    IMAGE_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',  
    shuffle=False
)


mobilenetv2_features = mobilenetv2_feature_model.predict(train_generator, steps=len(train_generator))
resnet50_features = resnet50_feature_model.predict(train_generator, steps=len(train_generator))


np.save(os.path.join(PROCESSED_DIR, 'mobilenetv2_train_features.npy'), mobilenetv2_features)
np.save(os.path.join(PROCESSED_DIR, 'resnet50_train_features.npy'), resnet50_features)


y_train = train_generator.classes
np.save(os.path.join(PROCESSED_DIR, 'y_train.npy'), y_train)


np.save(os.path.join(PROCESSED_SHAPE_DIR, 'mobilenetv2_shape.npy'), mobilenetv2_features.shape)
np.save(os.path.join(PROCESSED_SHAPE_DIR, 'resnet50_shape.npy'), resnet50_features.shape)

print(f"MobileNetV2 features shape: {mobilenetv2_features.shape}")
print(f"ResNet50 features shape: {resnet50_features.shape}")
print(f"Labels saved with shape: {y_train.shape}")
