# file: app.py
import cv2
import numpy as np
from mtcnn import MTCNN

# Import module tự viết
from face_verifier import FaceVerifier

# Import ResNet50 (Giữ nguyên cấu trúc thư mục của bạn)
from resnet50 import Build_ResNet50_VGGFace2
from load_weights import load_vggface2_weights

# CẤU HÌNH ĐƯỜNG DẪN
MLP_MODEL_PATH = 'mlp_face_classifier.h5'
ENCODER_PATH = 'label_encoder.pkl'
CENTROIDS_PATH = 'centroids.pkl'

def preprocess_vggface2(img):
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32')
    img[..., 0] -= 91.4953
    img[..., 1] -= 103.8827
    img[..., 2] -= 131.0912
    return img

def main():
    print("[INFO] Đang tải các mô hình (MTCNN, ResNet50)...")
    detector = MTCNN()
    
    # Khởi tạo ResNet50
    resnet_model = Build_ResNet50_VGGFace2()
    load_vggface2_weights(resnet_model)
    
    # Khởi tạo "Lớp bảo vệ 2 lớp" (MLP + Cosine)
    verifier = FaceVerifier(
        model_path=MLP_MODEL_PATH,
        encoder_path=ENCODER_PATH,
        centroids_path=CENTROIDS_PATH,
        cosine_threshold=0.6, # CÓ THỂ ĐIỀU CHỈNH: Tăng lên 0.65 hoặc 0.7 nếu vẫn nhận nhầm
        mlp_threshold=0.8 # CÓ THỂ ĐIỀU CHỈNH: Giảm xuống 0.8 để tăng độ nhạy, nhưng có thể tăng nhầm nhiều hơn
    )

    print("[INFO] Bật Webcam. Nhấn 'q' để thoát.")
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(rgb_frame)
        
        for res in results:
            if res['confidence'] < 0.90: continue
            
            x, y, w, h = res['box']
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(frame.shape[1], x+w), min(frame.shape[0], y+h)
            
            face_crop = rgb_frame[y1:y2, x1:x2]
            if face_crop.size == 0: continue
            
            # --- 1. RÚT TRÍCH VECTOR BẰNG RESNET50 ---
            face_preprocessed = preprocess_vggface2(face_crop)
            face_batch = np.expand_dims(face_preprocessed, axis=0)
            
            embedding = resnet_model.predict(face_batch, verbose=0)[0]
            embedding = embedding / np.linalg.norm(embedding) # L2 Normalize
            
            # --- 2. GỌI MODULE NHẬN DIỆN 2 LỚP ---
            # Chỉ cần 1 dòng code duy nhất!
            identity, mlp_prob, cosine_sim = verifier.identify(embedding)
            
            # --- 3. VẼ KẾT QUẢ ---
            if identity != "Unknown":
                color = (0, 255, 0) # Xanh lá cho người quen
                # Hiển thị cả % của MLP và độ tương đồng Cosine
                text = f"{identity} (MLP:{mlp_prob*100:.0f}%|Sim:{cosine_sim:.2f})"
            else:
                color = (0, 0, 255) # Đỏ cho người lạ
                # Hiển thị chỉ số để debug tại sao bị loại
                text = f"Unknown (Sim:{cosine_sim:.2f})"
                
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        cv2.imshow('Face Recognition System', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()