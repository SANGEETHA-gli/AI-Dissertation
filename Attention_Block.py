import tensorflow as tf
from tensorflow.keras import layers

def attention_block(inputs):
    # Compute attention scores
    attention = layers.Dense(1, activation='tanh')(inputs)
    attention = layers.Flatten()(attention)
    attention = layers.Activation('softmax')(attention)
    attention = layers.RepeatVector(inputs.shape[-1])(attention)
    attention = layers.Permute([2, 1])(attention)
    output_attention = layers.Multiply()([inputs, attention])
    
    return output_attention
