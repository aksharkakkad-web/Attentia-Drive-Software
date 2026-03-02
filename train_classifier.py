"""Train a binary distraction classifier on State Farm dataset.

Converts the 10-class State Farm Distracted Driver Detection dataset into
a binary task: c0 (safe driving) = attentive, c1-c9 = distracted.

Fine-tunes MobileNetV2 with a binary classification head.
Exports to TFLite with float16 quantization for edge deployment.

Usage:
    python train_classifier.py
"""

import os
import shutil
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR = Path("data/imgs/train")
BINARY_DIR = Path("data/binary")
MODEL_DIR = Path("models")
OUTPUT_TFLITE = MODEL_DIR / "classifier.tflite"
BACKUP_TFLITE = MODEL_DIR / "classifier_imagenet_backup.tflite"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.20
SEED = 42

# State Farm classes: c0 = safe, c1-c9 = distracted
ATTENTIVE_CLASSES = {"c0"}
DISTRACTED_CLASSES = {"c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9"}


def prepare_binary_dataset():
    """Reorganize State Farm data into binary folders: attentive / distracted."""
    for label in ("attentive", "distracted"):
        (BINARY_DIR / "train" / label).mkdir(parents=True, exist_ok=True)
        (BINARY_DIR / "val" / label).mkdir(parents=True, exist_ok=True)

    random.seed(SEED)

    stats = {"train_attentive": 0, "train_distracted": 0,
             "val_attentive": 0, "val_distracted": 0}

    for cls_dir in sorted(DATA_DIR.iterdir()):
        if not cls_dir.is_dir():
            continue

        cls_name = cls_dir.name
        if cls_name in ATTENTIVE_CLASSES:
            label = "attentive"
        elif cls_name in DISTRACTED_CLASSES:
            label = "distracted"
        else:
            continue

        images = sorted(cls_dir.glob("*.jpg"))
        random.shuffle(images)

        split_idx = int(len(images) * (1 - VAL_SPLIT))
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]

        for img in train_imgs:
            dst = BINARY_DIR / "train" / label / f"{cls_name}_{img.name}"
            if not dst.exists():
                shutil.copy2(img, dst)
            stats[f"train_{label}"] += 1

        for img in val_imgs:
            dst = BINARY_DIR / "val" / label / f"{cls_name}_{img.name}"
            if not dst.exists():
                shutil.copy2(img, dst)
            stats[f"val_{label}"] += 1

    print("Binary dataset prepared:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    return stats


def compute_class_weights(stats):
    """Compute class weights to handle attentive/distracted imbalance.

    Uses sklearn-style balanced weighting: total / (2 * n_class).
    Label 0 = attentive, label 1 = distracted.
    """
    n_att = stats["train_attentive"]
    n_dis = stats["train_distracted"]
    total = n_att + n_dis
    weights = {
        0: total / (2.0 * n_att),   # attentive gets higher weight
        1: total / (2.0 * n_dis),   # distracted gets lower weight
    }
    print(f"Class weights: attentive(0)={weights[0]:.3f}, distracted(1)={weights[1]:.3f}")
    return weights


def build_model():
    """Build MobileNetV2 with binary classification head."""
    base = keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )

    # Freeze base initially
    base.trainable = False

    model = keras.Sequential([
        base,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model, base


def create_datasets():
    """Create tf.data datasets with augmentation."""
    train_ds = keras.utils.image_dataset_from_directory(
        BINARY_DIR / "train",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        seed=SEED,
        class_names=["attentive", "distracted"],
    )

    val_ds = keras.utils.image_dataset_from_directory(
        BINARY_DIR / "val",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        seed=SEED,
        class_names=["attentive", "distracted"],
    )

    # Preprocessing: MobileNetV2 expects [-1, 1] range
    preprocess = keras.applications.mobilenet_v2.preprocess_input

    # Augmentation for training
    augment = keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomBrightness(0.2),
        keras.layers.RandomContrast(0.2),
    ])

    def prepare_train(image, label):
        image = preprocess(image)
        image = augment(image, training=True)
        return image, label

    def prepare_val(image, label):
        image = preprocess(image)
        return image, label

    train_ds = (train_ds
                .map(prepare_train, num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(tf.data.AUTOTUNE))

    val_ds = (val_ds
              .map(prepare_val, num_parallel_calls=tf.data.AUTOTUNE)
              .prefetch(tf.data.AUTOTUNE))

    return train_ds, val_ds


def convert_to_tflite(model):
    """Convert trained model to TFLite with float16 quantization."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()

    # Backup existing classifier if present
    if OUTPUT_TFLITE.exists() and not BACKUP_TFLITE.exists():
        shutil.copy2(OUTPUT_TFLITE, BACKUP_TFLITE)
        print(f"Backed up existing classifier to {BACKUP_TFLITE}")

    OUTPUT_TFLITE.write_bytes(tflite_model)
    size_mb = len(tflite_model) / 1e6
    print(f"TFLite model saved: {OUTPUT_TFLITE} ({size_mb:.1f} MB)")

    return tflite_model


def verify_tflite(tflite_path):
    """Quick verification of the exported TFLite model."""
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    print(f"TFLite verification:")
    print(f"  Input:  shape={inp['shape']} dtype={inp['dtype']}")
    print(f"  Output: shape={out['shape']} dtype={out['dtype']}")

    # Test with random input
    test_input = np.random.rand(1, *IMG_SIZE, 3).astype(np.float32)
    test_input = keras.applications.mobilenet_v2.preprocess_input(test_input)
    interpreter.set_tensor(inp["index"], test_input)
    interpreter.invoke()
    output = interpreter.get_tensor(out["index"])
    print(f"  Test output: {output.flatten()[0]:.4f} (should be ~0.5 for random input)")

    output_size = int(out["shape"][-1])
    if output_size <= 2:
        print(f"  Model type: BINARY ({output_size} output) — will be auto-detected correctly")
    else:
        print(f"  WARNING: {output_size} outputs — not binary!")


def main():
    print("=" * 60)
    print("Training Binary Distraction Classifier")
    print("=" * 60)

    # Step 1: Prepare binary dataset
    print("\n[1/5] Preparing binary dataset...")
    stats = prepare_binary_dataset()

    # Compute class weights to fix 1:8 attentive:distracted imbalance
    class_weights = compute_class_weights(stats)

    # Step 2: Create datasets
    print("\n[2/5] Creating data loaders...")
    train_ds, val_ds = create_datasets()

    # Step 3: Build and train (frozen base)
    print("\n[3/5] Training Phase 1: frozen base (transfer learning)...")
    model, base = build_model()
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=3,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, verbose=1,
        ),
    ]

    model.fit(
        train_ds, validation_data=val_ds,
        epochs=EPOCHS, callbacks=callbacks,
        class_weight=class_weights,
    )

    # Step 4: Fine-tune top layers of base
    print("\n[4/5] Training Phase 2: fine-tuning top layers...")
    base.trainable = True
    # Freeze everything except last 30 layers
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE / 10),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_ds, validation_data=val_ds,
        epochs=10, callbacks=callbacks,
        class_weight=class_weights,
    )

    # Step 5: Evaluate and export
    print("\n[5/5] Evaluating and exporting...")
    loss, acc = model.evaluate(val_ds)
    print(f"Final validation — Loss: {loss:.4f}, Accuracy: {acc:.4f}")

    tflite_model = convert_to_tflite(model)
    verify_tflite(OUTPUT_TFLITE)

    print("\n" + "=" * 60)
    print(f"Done! Model saved to {OUTPUT_TFLITE}")
    print(f"Validation accuracy: {acc:.1%}")
    print("The pipeline will auto-detect this as a binary classifier.")
    print("=" * 60)


if __name__ == "__main__":
    main()
