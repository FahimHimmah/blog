# ===============================
#  FINE-TUNING MOBILENETV2 UNTUK DETEKSI HAMA & NON-HAMA
# ===============================
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix, classification_report

# ===============================
# 1️⃣ LOAD & AUGMENT DATASET
# ===============================
train_dir = "dataset/train"
val_dir = "dataset/val"

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(224, 224),
    batch_size=32,
    label_mode='categorical'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=(224, 224),
    batch_size=32,
    label_mode='categorical'
)

class_names = train_ds.class_names
print(f"Kelas terdeteksi ({len(class_names)}):", class_names)

# Augmentasi data untuk memperkuat model
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
])

# Preprocessing khusus MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

AUTOTUNE = tf.data.AUTOTUNE
train_ds = (
    train_ds
    .map(lambda x, y: (data_augmentation(x), y))
    .map(lambda x, y: (preprocess_input(x), y))
    .cache()
    .shuffle(1000)
    .prefetch(AUTOTUNE)
)

val_ds = (
    val_ds
    .map(lambda x, y: (preprocess_input(x), y))
    .cache()
    .prefetch(AUTOTUNE)
)

# ===============================
# 2️⃣ BANGUN MODEL
# ===============================
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

# ===============================
# 3️⃣ TRAINING TAHAP 1: FEATURE EXTRACTION
# ===============================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath="checkpoint_mobilenetv2.keras",
    save_best_only=True,
    monitor='val_loss',
    verbose=1
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=2,
    min_lr=1e-7,
    verbose=1
)

print("=== Tahap 1: Training Feature Extractor ===")
history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[checkpoint, early_stop, reduce_lr],
    verbose=1
)

# ===============================
# 4️⃣ FINE-TUNING TAHAP 2
# ===============================
fine_tune_at = 120
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
for layer in base_model.layers[fine_tune_at:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("=== Tahap 2: Fine-Tuning MobileNetV2 ===")
history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=[checkpoint, early_stop, reduce_lr],
    verbose=1
)

# ===============================
# 5️⃣ SIMPAN MODEL
# ===============================
model.save("model_hama_dan_nonhama_mobilenetv2_final.keras")
print("✅ Model tersimpan: model_hama_dan_nonhama_mobilenetv2_final.keras")

# Simpan daftar label kelas
with open("class_labels.txt", "w") as f:
    for name in class_names:
        f.write(f"{name}\n")
print("✅ Label kelas disimpan ke class_labels.txt")
# ===============================
# 6️⃣ VISUALISASI HASIL TRAINING
# ===============================

# Gabungkan history tahap 1 dan tahap 2
loss_train = history1.history['loss'] + history2.history['loss']
loss_val = history1.history['val_loss'] + history2.history['val_loss']
acc_train = history1.history['accuracy'] + history2.history['accuracy']
acc_val = history1.history['val_accuracy'] + history2.history['val_accuracy']

# Hitung jumlah total epoch (untuk sumbu x)
epochs_total = range(1, len(loss_train) + 1)

plt.figure(figsize=(12,5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(epochs_total, loss_train, label='Train Loss')
plt.plot(epochs_total, loss_val, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()
plt.grid(True)

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs_total, acc_train, label='Train Accuracy')
plt.plot(epochs_total, acc_val, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Confusion Matrix (Absolute Counts)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
plt.close()
# ===============================
# 7️⃣ EVALUASI
# ===============================
y_true = np.concatenate([y for _, y in val_ds], axis=0)
y_pred = model.predict(val_ds)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_true, axis=1)

cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)

# 🔹 Confusion Matrix Normalized (persentase)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(10, 8))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Normalized)')
plt.show()
plt.close()

print("\n=== Classification Report ===")
print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))

# ===============================
