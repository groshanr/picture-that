import keras
from keras import layers, Sequential
import tensorflow as tf
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
import wandb

wandb.init()

# Create train dataset
train_split = keras.utils.image_dataset_from_directory(
    'Drawings',
    labels="inferred",
    label_mode="categorical",
    color_mode="rgb",
    batch_size=8,
    image_size=(500, 500),
    shuffle=True,
    seed=42,
    validation_split=0.2,
    subset='training',
    interpolation="bilinear",
)

# Create validation dataset
val_ds = keras.utils.image_dataset_from_directory(
    'Drawings',
    labels="inferred",
    label_mode="categorical",
    color_mode="rgb",
    batch_size=8,
    image_size=(500, 500),
    shuffle=True,
    seed=42,
    validation_split=0.2,
    subset='validation',
    interpolation="bilinear",
)

# Reference: https://www.tensorflow.org/tutorials/images/data_augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandAugment(
      value_range=(0, 255),
      num_ops=2,
      factor=0.5,
      interpolation="bilinear",
      seed=42
    ),
])

# Add augmented data to train dataset
train_ds = train_split.map(
  lambda x, y: (data_augmentation(x), y))

# Define model architecture
# Reference: https://www.tensorflow.org/tutorials/images/cnn
model = Sequential([
    layers.InputLayer(shape=(500,500,3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(2)
])

# compile the model
model.compile(loss='binary_crossentropy', metrics=['accuracy'])

# #https://wandb.ai/wandb_fc/articles/reports/Hyperparameter-Tuning-for-Keras-and-Pytorch-models--Vmlldzo1NDMyMzkx
model.fit(train_ds,  validation_data=(val_ds), callbacks=[WandbMetricsLogger(),
        WandbModelCheckpoint("models-aug.keras")])


model.save("cnn-aug.keras")




