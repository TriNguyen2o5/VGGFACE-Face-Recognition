from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers import BatchNormalization, Activation, Add, Flatten, Dense
from keras.models import Model
from keras import backend as K

# ======================================================================
# 1. KHỐI IDENTITY BLOCK (Đã sửa lỗi Naming Convention của Batch Norm)
# ======================================================================
def resnet_identity_block(input_tensor, kernel_size, filters, stage, block, bias=False):
    filters1, filters2, filters3 = filters
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    
    # Đặt tên chuẩn xác theo mã nguồn gốc của VGGFace2
    conv1_reduce_name = 'conv' + str(stage) + "_" + str(block) + "_1x1_reduce"
    conv1_increase_name = 'conv' + str(stage) + "_" + str(block) + "_1x1_increase"
    conv3_name = 'conv' + str(stage) + "_" + str(block) + "_3x3"

    x = Conv2D(filters1, (1, 1), use_bias=bias, name=conv1_reduce_name)(input_tensor)
    # Tên của BN phải là tên của Conv cộng thêm "/bn"
    x = BatchNormalization(axis=bn_axis, name=conv1_reduce_name + "/bn")(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', use_bias=bias, name=conv3_name)(x)
    x = BatchNormalization(axis=bn_axis, name=conv3_name + "/bn")(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), use_bias=bias, name=conv1_increase_name)(x)
    x = BatchNormalization(axis=bn_axis, name=conv1_increase_name + "/bn")(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    return x

# ======================================================================
# 2. KHỐI CONV BLOCK (Đã sửa lỗi Naming Convention của Batch Norm)
# ======================================================================
def resnet_conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), bias=False):
    filters1, filters2, filters3 = filters
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    
    conv1_reduce_name = 'conv' + str(stage) + "_" + str(block) + "_1x1_reduce"
    conv1_increase_name = 'conv' + str(stage) + "_" + str(block) + "_1x1_increase"
    conv1_proj_name = 'conv' + str(stage) + "_" + str(block) + "_1x1_proj"
    conv3_name = 'conv' + str(stage) + "_" + str(block) + "_3x3"

    x = Conv2D(filters1, (1, 1), strides=strides, use_bias=bias, name=conv1_reduce_name)(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=conv1_reduce_name + "/bn")(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', use_bias=bias, name=conv3_name)(x)
    x = BatchNormalization(axis=bn_axis, name=conv3_name + "/bn")(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv1_increase_name, use_bias=bias)(x)
    x = BatchNormalization(axis=bn_axis, name=conv1_increase_name + "/bn")(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides, use_bias=bias, name=conv1_proj_name)(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=conv1_proj_name + "/bn")(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

# ======================================================================
# 3. LẮP RÁP KIẾN TRÚC RESNET-50 HOÀN CHỈNH
# ======================================================================
def Build_ResNet50_VGGFace2(input_shape=(224, 224, 3)):
    img_input = Input(shape=input_shape)
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1

    # Block 1 (Stem): Conv 7x7 và MaxPool
    x = Conv2D(64, (7, 7), use_bias=False, strides=(2, 2), padding='same', name='conv1/7x7_s2')(img_input)
    x = BatchNormalization(axis=bn_axis, name='conv1/7x7_s2/bn')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # Block 2 (Stage 2)
    x = resnet_conv_block(x, 3, [64, 64, 256], stage=2, block=1, strides=(1, 1))
    x = resnet_identity_block(x, 3, [64, 64, 256], stage=2, block=2)
    x = resnet_identity_block(x, 3, [64, 64, 256], stage=2, block=3)

    # Block 3 (Stage 3)
    x = resnet_conv_block(x, 3, [128, 128, 512], stage=3, block=1)
    x = resnet_identity_block(x, 3, [128, 128, 512], stage=3, block=2)
    x = resnet_identity_block(x, 3, [128, 128, 512], stage=3, block=3)
    x = resnet_identity_block(x, 3, [128, 128, 512], stage=3, block=4)

    # Block 4 (Stage 4)
    x = resnet_conv_block(x, 3, [256, 256, 1024], stage=4, block=1)
    x = resnet_identity_block(x, 3, [256, 256, 1024], stage=4, block=2)
    x = resnet_identity_block(x, 3, [256, 256, 1024], stage=4, block=3)
    x = resnet_identity_block(x, 3, [256, 256, 1024], stage=4, block=4)
    x = resnet_identity_block(x, 3, [256, 256, 1024], stage=4, block=5)
    x = resnet_identity_block(x, 3, [256, 256, 1024], stage=4, block=6)

    # Block 5 (Stage 5)
    x = resnet_conv_block(x, 3, [512, 512, 2048], stage=5, block=1)
    x = resnet_identity_block(x, 3, [512, 512, 2048], stage=5, block=2)
    x = resnet_identity_block(x, 3, [512, 512, 2048], stage=5, block=3)

    # ==================================================================
    # Đuôi mạng chuẩn của VGGFace2
    # ==================================================================
    x = AveragePooling2D((7, 7), name='avg_pool')(x)
    x = GlobalAveragePooling2D()(x)

    model = Model(img_input, x, name='vggface_resnet50_custom')
    return model

# Test thử khởi tạo
if __name__ == '__main__':
    my_model = Build_ResNet50_VGGFace2()
    print("Khởi tạo thành công mô hình ResNet50 từ đầu!")
    my_model.summary()