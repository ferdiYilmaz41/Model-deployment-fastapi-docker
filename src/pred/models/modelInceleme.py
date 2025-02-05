import h5py
import os
import tensorflow as tf
model_path = os.path.join(os.path.dirname(__file__), 'model.h5')
model= tf.keras.models.load_model(model_path)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.compile_metrics())