"""
@author: Kamaleshwar M

"""
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Input, BatchNormalization, Dropout, LayerNormalization

def create_model(input_shape: tuple) -> Model:

    inputs = Input(shape=input_shape)
    x = Bidirectional(LSTM())(inputs)
    x = LayerNormalization()(x)
    x = Dropout()(x)
    x = Bidirectional(LSTM())(x)
    x = LayerNormalization()(x)
    x = Dropout()(x)
    x = Bidirectional(LSTM())(x)
    x = LayerNormalization()(x)
    x = Dropout()(x)
    outputs = Dense()(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=,
        beta_1=,
        beta_2=,
        clipnorm=
    )
    model.compile(
        optimizer=optimizer,
        loss=, 
        metrics=['mae', 'accuracy']
    )
    return model