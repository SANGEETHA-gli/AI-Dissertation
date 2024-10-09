import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, fbeta_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from attention_block import attention_block  # Assuming you already have this
from transformer_block import TransformerBlock  # Assuming you already have this

# Enable mixed precision training using the updated API
from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')  # Use mixed precision to optimize memory usage

# Paths
PROCESSED_DIR = r'C:\Users\dsang\OneDrive\Desktop\mindwell_backend\mindwell_backend\data\processedaffectnet'
MODEL_SAVE_PATH = r'C:\Users\dsang\OneDrive\Desktop\mindwell_backend\mindwell_backend\models\models\hybrid_mobilenet_resnet_model_dup.h5'  # New save path

# Load extracted features and labels
mobilenetv2_features = np.load(os.path.join(PROCESSED_DIR, 'mobilenetv2_train_features.npy'))
resnet50_features = np.load(os.path.join(PROCESSED_DIR, 'resnet50_train_features.npy'))
y_train = np.load(os.path.join(PROCESSED_DIR, 'y_train.npy'))

# Concatenate the MobileNetV2 and ResNet50 features
combined_features = np.concatenate([mobilenetv2_features, resnet50_features], axis=1)

# Ensure labels are binary (0 or 1)
y_train = np.where(y_train > 0, 1, 0)  # Map non-zero values to 1 (depression), zero to 0 (non-depression)

# Apply Random Oversampling to balance the classes
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(combined_features, y_train)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_resampled, y_train_resampled, test_size=0.2, random_state=42)

# Class Weights: Give higher weight to non-depression to account for imbalance in the validation set
class_weights = {
    0: 3.0,  # Increase the weight for non-depression further
    1: 0.5   # Reduce weight for depression class slightly
}

# Update TransformerBlock to add get_config for saving
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate
        })
        return config


# Build the hybrid model with attention, transformers, and feature compression
def build_hybrid_model(input_shape):
    inputs = layers.Input(shape=input_shape)

    # Compress features to reduce memory usage before applying attention and transformers
    x = layers.Dense(1024, activation='relu')(inputs)

    # Apply attention mechanism
    attention_output = attention_block(x)

    # Apply transformer block (embed_dim=1024 to match compressed feature size)
    transformer_layer = TransformerBlock(embed_dim=1024, num_heads=4, ff_dim=512)
    transformer_output = transformer_layer(attention_output)

    # Flatten and fully connected layers for classification
    x = layers.Flatten()(transformer_output)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)  # Binary classification for depression

    model = models.Model(inputs=inputs, outputs=outputs)
    return model

# Build and compile the model
model = build_hybrid_model(input_shape=(combined_features.shape[1],))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),  # Lower learning rate for better convergence
              loss='binary_crossentropy',  # Using binary_crossentropy since we are working with binary labels (0/1)
              metrics=['accuracy'])

# Callbacks for early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# Train the model with class weights, oversampled data, and smaller batch size (to avoid OOM error)
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=8,  # Reduced batch size to prevent OOM issues
    class_weight=class_weights,  # Apply class weights
    callbacks=[early_stopping, lr_scheduler]
)

# Save the trained hybrid model
model.save(MODEL_SAVE_PATH)
print(f"Hybrid model saved to: {MODEL_SAVE_PATH}")

# ---- Model Evaluation with Threshold Adjustment ----

# Evaluate the model
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")

# Predictions on validation set with adjusted threshold
threshold = 0.2  # Lower threshold
y_pred = (model.predict(X_val) > threshold).astype('int32')

# Classification report
print("Classification Report with adjusted class weights, oversampling, and threshold:")
print(classification_report(y_val, y_pred, target_names=['Non-Depression', 'Depression']))

# Precision, Recall, and F2 Score
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f2_score = fbeta_score(y_val, y_pred, beta=2)  # F2 score with more focus on recall

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F2 Score: {f2_score}")
