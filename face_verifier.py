import pickle
import numpy as np
from keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity

class FaceVerifier:
    def __init__(self, model_path, encoder_path, centroids_path, cosine_threshold=0.6, mlp_threshold=0.9):
        """Khởi tạo module, nạp model và các bộ nhớ"""
        print("[INFO] Đang khởi tạo bộ giải mã FaceVerifier...")
        self.model = load_model(model_path)
        self.cosine_threshold = cosine_threshold
        self.mlp_threshold = mlp_threshold
        
        with open(encoder_path, 'rb') as f:
            self.le = pickle.load(f)
            
        with open(centroids_path, 'rb') as f:
            self.centroids = pickle.load(f)
        print("[INFO] FaceVerifier đã sẵn sàng!")

    def identify(self, face_vector):
        """
        Nhận vào 1 vector khuôn mặt (2048 chiều)
        Trả về: Tên người (hoặc Unknown), độ tự tin của MLP, và độ tương đồng Cosine
        """
        vec = face_vector.reshape(1, -1)
        
        # 1. MLP Dự đoán (Lớp 1)
        probs = self.model.predict(vec, verbose=0)
        pred_idx = np.argmax(probs)
        mlp_conf = probs[0][pred_idx]
        
        # 2. Kiểm tra lại bằng Cosine Similarity (Lớp 2)
        centroid = self.centroids[pred_idx].reshape(1, -1)
        sim = cosine_similarity(vec, centroid)[0][0]
        
        name = self.le.classes_[pred_idx]
        
        # 3. Ra quyết định dựa trên cả 2 ngưỡng
        if mlp_conf >= self.mlp_threshold and sim >= self.cosine_threshold:
            return name, mlp_conf, sim
        else:
            return "Unknown", mlp_conf, sim