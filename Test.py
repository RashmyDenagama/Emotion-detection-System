from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import os

train_data_dir = r"C:\Users\Rashmi Denagama\Desktop\FacialRecognitionAI\Data\train"
validation_data_dir = r"C:\Users\Rashmi Denagama\Desktop\FacialRecognitionAI\Data\test"

# Data Augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Normalization for validation data
validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Load training and validation data using flow_from_directory
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

# Emotion Labels
class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Build CNN model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))  # Dropout to prevent overfitting

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))  # Dropout to prevent overfitting

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))  # Dropout to prevent overfitting

model.add(Flatten())  # Flatten the feature maps

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))  # Prevent overfitting with dropout

model.add(Dense(7, activation='softmax'))  # 7 classes for emotion labels

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary to verify
print(model.summary())

# Calculate number of images for steps_per_epoch and validation_steps
num_train_imgs = sum([len(files) for _, _, files in os.walk(train_data_dir)])
num_test_imgs = sum([len(files) for _, _, files in os.walk(validation_data_dir)])

print(f"Training images: {num_train_imgs}")
print(f"Test images: {num_test_imgs}")

# Set number of epochs for training
epochs = 30

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=num_train_imgs // 32,  # Number of batches per epoch
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=num_test_imgs // 32  # Number of batches per validation
)

# Save the trained model
model.save('emotion_detection_model.h5')
