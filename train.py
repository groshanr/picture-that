import keras
from keras import layers, Sequential
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
import wandb

wandb.init()

# Create train dataset
train_ds = keras.utils.image_dataset_from_directory(
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

# Compile and train the model
model.compile(loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_ds,  validation_data=(val_ds), callbacks=[WandbMetricsLogger(),
        WandbModelCheckpoint("models.keras")])

model.save("cnn.keras")




