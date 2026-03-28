import os
import cv2
import numpy as np
import pickle
from tqdm import tqdm

from resnet50 import Build_ResNet50_VGGFace2
from load_weights import load_vggface2_weights

DATASET_DIR = 'dataset_cropped'
OUTPUT_DB = 'cosine_face_db.pkl'

def preprocess_vggface2(img):
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32')
    img[..., 0] -= 91.4953
    img[..., 1] -= 103.8827
    img[..., 2] -= 131.0912
    return img

def main():
    print("[INFO] Khởi tạo ResNet50-VGGFace2...")
    model = Build_ResNet50_VGGFace2()
    load_vggface2_weights(model)
    
    # Dùng Dictionary để lưu cho tiện: { 'do': [vec1, vec2...], 'lam': [vec1...], 'tri': [vec1...] }
    database = {} 

    classes = os.listdir(DATASET_DIR)
    
    for person_name in classes:
        person_dir = os.path.join(DATASET_DIR, person_name)
        if not os.path.isdir(person_dir): continue
            
        print(f"-> Đang trích xuất Vector cho: {person_name}")
        vectors = []
        
        for img_name in tqdm(os.listdir(person_dir)):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path)
            if img is None: continue
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Đưa qua Backbone lấy Vector
            vec = model.predict(np.expand_dims(preprocess_vggface2(img_rgb), axis=0), verbose=0)[0]
            
            # CHUẨN HÓA L2 (Bắt buộc cho thuật toán Cosine)
            vec = vec / np.linalg.norm(vec) 
            vectors.append(vec)
            
        database[person_name] = vectors

    with open(OUTPUT_DB, 'wb') as f:
        pickle.dump(database, f)
        
    print(f"\n[THÀNH CÔNG] Đã lưu Database {len(database)} người vào {OUTPUT_DB}")
    print("=> GIỜ BẠN CÓ THỂ MỞ WEBCAM LÊN NHẬN DIỆN NGAY, KHÔNG CẦN TRAIN!")

if __name__ == '__main__':
    main()