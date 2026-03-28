import os
from typing import List
import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
from sklearn.utils import shuffle
from resnet50 import Build_ResNet50_VGGFace2
from load_weights import load_vggface2_weights

TRAIN_DIR = 'output_cropped_faces/train'
TEST_DIR = 'output_cropped_faces/test'
OUTPUT_DB = 'face_vectors_db_strict.pkl' # Đổi tên file db để phân biệt
AUGMENT_FACTOR = 5 # 1 ảnh gốc sinh ra 5 ảnh biến thể (chỉ áp dụng cho tập Train)

def preprocess_vggface2(img):
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32')
    img[..., 0] -= 91.4953
    img[..., 1] -= 103.8827
    img[..., 2] -= 131.0912
    return img

def process_dataset(data_dir, model, augmenter=None, is_train=True):
    X, y = [], []

    classes = os.listdir(data_dir)

    for person_name in classes:
        person_dir = os.path.join(data_dir, person_name)
        if not os.path.isdir(person_dir): continue

        image_files = [os.path.join(person_dir, f) for f in os.listdir(person_dir)]

        print(f"[INFO] {person_name} - {len(image_files)} ảnh")

        for img_path in tqdm(image_files):
            img = cv2.imread(img_path)
            if img is None: continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # --- Ảnh gốc ---
            vec = model.predict(
                np.expand_dims(preprocess_vggface2(img_rgb), axis=0),
                verbose=0
            )[0]

            X.append(vec / np.linalg.norm(vec))
            y.append(person_name)

            # --- Augmentation chỉ cho train ---
            if is_train and augmenter is not None:
                count = 0
                for batch in augmenter.flow(np.expand_dims(img_rgb, axis=0), batch_size=1):
                    aug_img = batch[0].astype('uint8')

                    aug_vec = model.predict(
                        np.expand_dims(preprocess_vggface2(aug_img), axis=0),
                        verbose=0
                    )[0]

                    X.append(aug_vec / np.linalg.norm(aug_vec))
                    y.append(person_name)

                    count += 1
                    if count >= AUGMENT_FACTOR:
                        break

    return X, y

def main():
    print("[INFO] Load model VGGFace2...")
    model = Build_ResNet50_VGGFace2()
    load_vggface2_weights(model)

    # Augmentation nhẹ hơn
    augmenter = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        brightness_range=[0.9, 1.1],
        horizontal_flip=True
    )

    X_train: List[np.ndarray]
    y_train: List[str]

    print("\n[INFO] ===== PROCESS TRAIN =====")
    X_train, y_train = process_dataset(TRAIN_DIR, model, augmenter, is_train=True)

    print("\n[INFO] ===== PROCESS TEST =====")
    X_test, y_test = process_dataset(TEST_DIR, model, augmenter=None, is_train=False)

    print(f"\n[INFO] Train: {len(X_train)} | Test: {len(X_test)}")
    indices = np.arange(len(X_train))
    np.random.seed(42)
    np.random.shuffle(indices)

    X_train = [X_train[i] for i in indices]
    y_train = [y_train[i] for i in indices]

    with open(OUTPUT_DB, 'wb') as f:
        pickle.dump((X_train, y_train, X_test, y_test), f)

    print(f"[SUCCESS] Saved to {OUTPUT_DB}")


if __name__ == '__main__':
    main()