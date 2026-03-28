import os
import numpy as np
from keras.utils.data_utils import get_file
from resnet50 import Build_ResNet50_VGGFace2  # Import file kiến trúc bạn vừa code ở bước trước

def load_vggface2_weights(model):
    """
    Hàm này lấy file weights chuẩn gốc của VGGFace2 (ResNet50_no_top) 
    và nạp vào kiến trúc build from scratch của bạn.
    """
    # Link chính thức từ repo keras-vggface của rcmalli
    TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_notop_resnet50.h5'
    
    print("1. Đang kiểm tra / tải file trọng số (Weights) gốc từ Github...")
    # get_file sẽ tự động tải nếu máy bạn chưa có, mặc định lưu vào ~/.keras/models/vggface/
    weights_path = get_file('rcmalli_vggface_tf_notop_resnet50.h5',
                            TF_WEIGHTS_PATH_NO_TOP,
                            cache_subdir='models/vggface')
    
    print(f"   -> File trọng số nằm tại: {weights_path}")
    print("2. Đang nạp trọng số vào kiến trúc ResNet50 của bạn...")
    
    # ĐÂY LÀ DÒNG CODE "ĂN TIỀN" NHẤT: by_name=True
    # Nó sẽ dò tìm từng layer có TÊN giống hệt nhau giữa file .h5 và code của bạn để bơm số liệu vào.
    model.load_weights(weights_path, by_name=True)
    print("   -> Nạp trọng số THÀNH CÔNG! 'Vỏ' kiến trúc đã có 'Não' bên trong.")

if __name__ == '__main__':
    # BƯỚC 1: Khởi tạo mô hình rỗng
    print("Khởi tạo cấu trúc mạng ResNet-50...")
    my_model = Build_ResNet50_VGGFace2(input_shape=(224, 224, 3))
    
    # BƯỚC 2: Gọi hàm nạp tạ
    load_vggface2_weights(my_model)
    
    # BƯỚC 3: KIỂM CHỨNG LUỒNG DỮ LIỆU (Sanity Check)
    # Ta sẽ tạo 1 bức ảnh giả (Dummy Image) chứa các số ngẫu nhiên để xem mạng có chạy mượt không
    print("\n--- KIỂM CHỨNG HOẠT ĐỘNG INFERENCE ---")
    dummy_image = np.random.rand(1, 224, 224, 3).astype('float32') # shape: (batch_size, H, W, Channels)
    
    output_vector = my_model.predict(dummy_image)
    
    # Nếu trả về list, lấy phần tử đầu tiên
    if isinstance(output_vector, list):
        output_vector = output_vector[0]

    print(f"Input shape : {dummy_image.shape}  -> Tượng trưng cho 1 tấm ảnh 224x224 RGB")
    print(f"Output shape: {output_vector.shape} -> Tượng trưng cho Vector Embedding trích xuất được")
    
    if output_vector.shape == (1, 2048):
        print("=> XUẤT SẮC: Kích thước Vector chuẩn 2048 chiều của ResNet-50! Hệ thống Feature Extractor đã sẵn sàng.")
    else:
        print("=> CÓ LỖI: Kích thước Vector đầu ra không đúng chuẩn.")