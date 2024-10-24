import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, fbeta_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from attention_block import attention_block  
from transformer_block import TransformerBlock  

from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16') 


PROCESSED_DIR = r'C:\Users\dsang\OneDrive\Desktop\mindwell_backend\mindwell_backend\data\processedaffectnet'
MODEL_SAVE_PATH = r'C:\Users\dsang\OneDrive\Desktop\mindwell_backend\mindwell_backend\models\models\hybrid_mobilenet_resnet_model_dup.h5'  # New save path

mobilenetv2_features = np.load(os.path.join(PROCESSED_DIR, 'mobilenetv2_train_features.npy'))
resnet50_features = np.load(os.path.join(PROCESSED_DIR, 'resnet50_train_features.npy'))
y_train = np.load(os.path.join(PROCESSED_DIR, 'y_train.npy'))

combined_features = np.concatenate([mobilenetv2_features, resnet50_features], axis=1)

y_train = np.where(y_train > 0, 1, 0)  


ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(combined_features, y_train)


X_train, X_val, y_train, y_val = train_test_split(X_train_resampled, y_train_resampled, test_size=0.2, random_state=42)


class_weights = {
    0: 3.0, 
    1: 0.5   
}


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



def build_hybrid_model(input_shape):
    inputs = layers.Input(shape=input_shape)

   
    x = layers.Dense(1024, activation='relu')(inputs)

  
    attention_output = attention_block(x)

   
    transformer_layer = TransformerBlock(embed_dim=1024, num_heads=4, ff_dim=512)
    transformer_output = transformer_layer(attention_output)


    x = layers.Flatten()(transformer_output)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)  
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

# Build and compile the model
model = build_hybrid_model(input_shape=(combined_features.shape[1],))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),  
              loss='binary_crossentropy',  
              metrics=['accuracy'])


early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)


history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=8,  
    class_weight=class_weights, 
    callbacks=[early_stopping, lr_scheduler]
)


model.save(MODEL_SAVE_PATH)
print(f"Hybrid model saved to: {MODEL_SAVE_PATH}")



val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")


threshold = 0.2  
y_pred = (model.predict(X_val) > threshold).astype('int32')

print("Classification Report with adjusted class weights, oversampling, and threshold:")
print(classification_report(y_val, y_pred, target_names=['Non-Depression', 'Depression']))

precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f2_score = fbeta_score(y_val, y_pred, beta=2)  

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F2 Score: {f2_score}")
