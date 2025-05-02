import keras
from keras import layers, Sequential
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
import wandb
from keras.applications.resnet50 import ResNet50
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense, Flatten

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

# Augment training data
# Source: https://www.tensorflow.org/tutorials/images/data_augmentation
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

# Reference: https://medium.com/@kenneth.ca95/a-guide-to-transfer-learning-with-keras-using-resnet50-a81a4a28084b
# Load pre-trained model without top layer
pre_model = ResNet50(weights='imagenet', include_top=False, input_shape=(500, 500, 3))

# Freeze the weights of the pre-trained layers
for layer in pre_model.layers:
    layer.trainable = False

# Create new output layer
x = Flatten()(pre_model.output)
x = Dense(32, activation='relu')(x)
x = Dense(32, activation='relu')(x)
x = Dense(32, activation='relu')(x)
output_layer = Dense(2, activation='softmax')(x)

# Create finetuned model
fine_model = Model(inputs=pre_model.input, outputs=output_layer)

# Display model summary
fine_model.summary()

# Compile & train model
fine_model.compile(loss='binary_crossentropy', metrics=['accuracy'])

trained_fine_model = fine_model.fit(train_ds, validation_data=(val_ds), callbacks=[WandbMetricsLogger(),
        WandbModelCheckpoint("ft-resnet50-aug.keras")])









