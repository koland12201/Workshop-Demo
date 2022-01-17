import tensorflow as tf
import matplotlib.pyplot as plt
import datetime

# training parameters
batch_size = 70
epochs = 40
IMG_HEIGHT = 256
IMG_WIDTH = 256


# print number of available GPUs (optional)
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # uncomment if you wish to remain training on CPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# configure training progress logging for tensorboard (optional)
# to visualize training data, do "tensorboard --logdir logs" and open "http://localhost:6006/"
log_dir = "logs\\" + datetime.datetime.now().strftime("%Y %m %d-%H %M %S")
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# import training data
train_dir = "Images"

# configure image generator with augmentation parameters
train_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                           horizontal_flip=True,
                                           vertical_flip=True,
                                           rotation_range=10,
                                           shear_range=0.03,
                                           width_shift_range=0.05,
                                           height_shift_range=0.05,
                                           zoom_range=0.03,
                                           validation_split=0.05)


# Import image to generator
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT,IMG_WIDTH),
                                                           class_mode='binary',
                                                           subset='training')

# validation gen
validation_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                                 directory=train_dir,
                                                                 target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                                 class_mode='binary',
                                                                 subset='validation')

# Randomly select 5 images to plot (optional)
def plotimages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

sample_training_images, _ = next(train_data_gen)
plotimages(sample_training_images[:5])

# create layers
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
model.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3)))
model.add(tf.keras.layers.Dropout(0.1))

model.add(tf.keras.layers.Conv2D(32, (3, 3),  padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3)))
model.add(tf.keras.layers.Dropout(0.1))

model.add(tf.keras.layers.Conv2D(64, (3, 3),  padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3)))
model.add(tf.keras.layers.Dropout(0.1))

model.add(tf.keras.layers.Conv2D(128, (3, 3),  padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3)))
model.add(tf.keras.layers.Dropout(0.1))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(2))

# Configure exponential decay learn rate
initial_learning_rate = 0.002
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=20,
    decay_rate=0.97,
    staircase=True)

# Configure optimizer
optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Compile model
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=optimizer,
              metrics=['accuracy'])
model.summary()

# Start training
model.fit(train_data_gen,
          validation_data=validation_data_gen,
          epochs=epochs,
          verbose=1,
          shuffle=1,
          callbacks=[tensorboard])

# Save trained model
model.save('Trained_model/model')
print("Model Saved!")