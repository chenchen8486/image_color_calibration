import numpy as np
import pandas as pd
import joblib, cv2, os, time
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from enum import Enum
from sklearn.preprocessing import MinMaxScaler


# ==================== 核心工具函数：中文路径支持 ====================
def imread_cn(path):
    """
    读取包含中文路径/文件名的图像
    :param path: 图像路径（可含中文）
    :return: BGR 格式的 numpy 数组，失败返回 None
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"读取图片失败: {path}, 错误: {e}")
        return None


# ================================================================

def load_data(file_path_, train_data_key_dict, sheet_name='sheet1'):
    _, ext = os.path.splitext(file_path_.lower())
    if ext == '.xlsx':
        engine = 'openpyxl'
    elif ext == '.xls':
        engine = 'xlrd'
    else:
        raise ValueError(f"不支持的文件格式: {ext}")
    try:
        df = pd.read_excel(file_path_, engine=engine, sheet_name=sheet_name)
    except Exception as e:
        raise RuntimeError(f"无法读取 Excel 文件 {file_path_}，请检查格式和依赖库。错误: {e}") from e

    column_names = df.columns
    data_dict = {}
    for column in column_names:
        temp = df[column].dropna().tolist()
        data_dict[column] = np.array(temp)

    input1 = train_data_key_dict['input1']
    input2 = train_data_key_dict['input2']
    input3 = train_data_key_dict['input3']
    output1 = train_data_key_dict['output1']
    output2 = train_data_key_dict['output2']
    output3 = train_data_key_dict['output3']

    n_samples = len(data_dict[input1])
    rgb_input = np.array([
        [data_dict[input1][i], data_dict[input2][i], data_dict[input3][i]]
        for i in range(n_samples)
    ])
    lab_output = np.array([
        [data_dict[output1][i], data_dict[output2][i], float(data_dict[output3][i])]
        for i in range(n_samples)
    ])
    return rgb_input, lab_output


def trainer(input_data, output_data, hidden_layer=(10, 10), max_iter=10000):
    X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.15, random_state=42)
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_L = MinMaxScaler(feature_range=(0, 1))
    scaler_a = MinMaxScaler(feature_range=(-1, 1))
    scaler_b = MinMaxScaler(feature_range=(-1, 1))

    dummy_rgb = np.array([[0, 0, 0], [255, 255, 255]])
    scaler_X.partial_fit(dummy_rgb)
    scaler_L.partial_fit(np.array([[0], [100]]))
    scaler_a.partial_fit(np.array([[-128], [127]]))
    scaler_b.partial_fit(np.array([[-128], [127]]))

    X_train_scaled = scaler_X.transform(X_train.astype(np.float64))
    X_test_scaled = scaler_X.transform(X_test.astype(np.float64))

    y_train_scaled_L = scaler_L.transform(y_train[:, 0].reshape(-1, 1).astype(np.float64))
    y_train_scaled_a = scaler_a.transform(y_train[:, 1].reshape(-1, 1).astype(np.float64))
    y_train_scaled_b = scaler_b.transform(y_train[:, 2].reshape(-1, 1).astype(np.float64))

    y_test_scaled_L = scaler_L.transform(y_test[:, 0].reshape(-1, 1).astype(np.float64))
    y_test_scaled_a = scaler_a.transform(y_test[:, 1].reshape(-1, 1).astype(np.float64))
    y_test_scaled_b = scaler_b.transform(y_test[:, 2].reshape(-1, 1).astype(np.float64))

    y_train_scaled = np.hstack([y_train_scaled_L, y_train_scaled_a, y_train_scaled_b])
    y_test_scaled = np.hstack([y_test_scaled_L, y_test_scaled_a, y_test_scaled_b])

    mlp_regressor = MLPRegressor(
        hidden_layer_sizes=hidden_layer,
        max_iter=max_iter,
        shuffle=True,
        random_state=42,
        solver='lbfgs',
        activation='tanh'
    )
    mlp_regressor.fit(X_train_scaled, y_train_scaled)

    train_score = mlp_regressor.score(X_train_scaled, y_train_scaled)
    test_score = mlp_regressor.score(X_test_scaled, y_test_scaled)
    print(f"\n训练集得分: {train_score:.4f}")
    print(f"测试集得分: {test_score:.4f}")

    return mlp_regressor, scaler_X, scaler_L, scaler_a, scaler_b


def save_model(save_root_, mlp_regressor_, scaler_X_, scaler_L_, scaler_a_, scaler_b_):
    joblib.dump(mlp_regressor_, os.path.join(save_root_, 'mlp_regressor_model.pkl'))
    joblib.dump(scaler_X_, os.path.join(save_root_, 'scaler_X.pkl'))
    joblib.dump(scaler_L_, os.path.join(save_root_, 'scaler_L.pkl'))
    joblib.dump(scaler_a_, os.path.join(save_root_, 'scaler_a.pkl'))
    joblib.dump(scaler_b_, os.path.join(save_root_, 'scaler_b.pkl'))


def load_model(load_root_):
    mlp_regressor = joblib.load(os.path.join(load_root_, 'mlp_regressor_model.pkl'))
    scaler_X = joblib.load(os.path.join(load_root_, 'scaler_X.pkl'))
    scaler_L = joblib.load(os.path.join(load_root_, 'scaler_L.pkl'))
    scaler_a = joblib.load(os.path.join(load_root_, 'scaler_a.pkl'))
    scaler_b = joblib.load(os.path.join(load_root_, 'scaler_b.pkl'))
    return mlp_regressor, scaler_X, scaler_L, scaler_a, scaler_b


def generate_rgb_to_lab_lut(mlp_regressor, scaler_X, scaler_L, scaler_a, scaler_b, save_root, bit_num=8,
                            linear_gain=0.90, highlight_contrast=1.3, saturation_gain=1.1):
    gray_scale = pow(2, bit_num)
    step = 255 / (gray_scale - 1)
    lookup_table_lab = np.zeros((gray_scale, gray_scale, gray_scale, 3), dtype=np.float64)
    lookup_table_srgb = np.zeros((gray_scale, gray_scale, gray_scale, 3), dtype=np.uint8)

    # 构造所有 RGB 组合 (N, 3)
    rgb_values_scaled = np.array([(r * step, g * step, b * step)
                                  for r in range(gray_scale)
                                  for g in range(gray_scale)
                                  for b in range(gray_scale)], dtype=np.float64)

    rgb_scaled = scaler_X.transform(rgb_values_scaled)
    predicted_lab_scaled = mlp_regressor.predict(rgb_scaled)

    lab_values = np.hstack([
        scaler_L.inverse_transform(predicted_lab_scaled[:, 0].reshape(-1, 1)),
        scaler_a.inverse_transform(predicted_lab_scaled[:, 1].reshape(-1, 1)),
        scaler_b.inverse_transform(predicted_lab_scaled[:, 2].reshape(-1, 1))
    ])

    lab_values[:, 0] = np.clip(lab_values[:, 0], 0, 100)
    lab_values[:, 1] = np.clip(lab_values[:, 1], -128, 127)
    lab_values[:, 2] = np.clip(lab_values[:, 2], -128, 127)

    # 转换为 sRGB (此时 lab_values 是 (N, 3) 的二维数组)
    srgb_values = convert_lab_to_srgb(lab_values, linear_gain=linear_gain,
                                      highlight_contrast=highlight_contrast, saturation_gain=saturation_gain)

    index = 0
    for r in range(gray_scale):
        for g in range(gray_scale):
            for b in range(gray_scale):
                lookup_table_lab[r, g, b] = lab_values[index]
                lookup_table_srgb[r, g, b] = srgb_values[index]
                index += 1

    lookup_table_lab_flat = lookup_table_lab.reshape(-1, 3).astype(np.float32)
    lookup_table_srgb_flat = lookup_table_srgb.reshape(-1, 3).astype(np.uint8)

    lookup_table_lab_flat.tofile(os.path.join(save_root, "rgb_to_lab_lookup_table.bin"))
    lookup_table_srgb_flat.tofile(os.path.join(save_root, "rgb_to_srgb.bin"))


def load_lookup_table_bin(file_path="rgb_to_lab_lookup_table.bin", dtype=np.float32, bit_num=8):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"\n找不到文件 {file_path}")
    data = np.fromfile(file_path, dtype=dtype)
    gray_scale = pow(2, bit_num)
    expected_size = gray_scale * gray_scale * gray_scale * 3
    if data.size != expected_size:
        raise ValueError(f"文件大小不匹配：期望 {expected_size} 个元素，实际有 {data.size} 个。")
    return data.reshape((gray_scale, gray_scale, gray_scale, 3))


def lab_to_xyz(lab):
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200
    xyz = np.zeros_like(lab, dtype=np.float64)
    epsilon = 0.008856
    kappa = 903.3
    xyz[..., 0] = np.where(fx ** 3 > epsilon, fx ** 3, (116 * fx - 16) / kappa) * 0.95047
    xyz[..., 1] = np.where(L > kappa * epsilon, ((L + 16) / 116) ** 3, L / kappa)
    xyz[..., 2] = np.where(fz ** 3 > epsilon, fz ** 3, (116 * fz - 16) / kappa) * 1.08883
    return xyz


def xyz_to_linear_rgb(xyz):
    M = np.array([
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252]
    ])
    linear_rgb = np.dot(xyz, M.T)
    return np.clip(linear_rgb, 0, None)


def linear_rgb_to_srgb(linear_rgb):
    srgb = np.where(
        linear_rgb <= 0.0031308,
        12.92 * linear_rgb,
        1.055 * (linear_rgb ** (1 / 2.4)) - 0.055
    )
    return np.clip(srgb, 0, 1)


# 2026.2.5修改：注销以下函数，临时把函数备份替换成后面函数
def convert_lab_to_srgb_bak(lab_image):
    """
    【修改版】为了适配现场不可变的 C++ 代码
    强制返回 RGB 格式的数据，而不是 BGR。
    """
    lab_image = lab_image.astype(np.float64)
    if np.max(lab_image[..., 0]) > 100 or np.min(lab_image[..., 0]) < 0:
        lab_image[..., 0] = np.clip(lab_image[..., 0], 0, 100)

    xyz = lab_to_xyz(lab_image)
    linear_rgb = xyz_to_linear_rgb(xyz)
    srgb = linear_rgb_to_srgb(linear_rgb)

    # 保留新版本的精度优化（四舍五入）
    srgb_uint8 = (srgb * 255 + 0.5).astype(np.uint8)

    # ================= 核心修改区域 =================
    # 原代码（新版本逻辑）：会自动转成 BGR，导致 C++ 颜色不对
    # if srgb_uint8.ndim == 2 and srgb_uint8.shape[1] == 3:
    #     return srgb_uint8[:, ::-1]  # RGB -> BGR
    # else:
    #     return cv2.cvtColor(srgb_uint8, cv2.COLOR_RGB2BGR)

    # 现代码（兼容老 C++ 版本逻辑）：
    # 既然 C++ 端会自己做转换，我们这里就保持 RGB 原样输出！
    return srgb_uint8
    # ===============================================



# 2026.2.5新增
def lab_to_lch(lab):
    """ 将 LAB 转换为 LCh 空间 (Lightness, Chroma, Hue) """
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    C = np.sqrt(a ** 2 + b ** 2)
    h = np.degrees(np.arctan2(b, a)) % 360
    return np.stack([L, C, h], axis=-1)


# 2026.2.5新增
def lch_to_lab(lch):
    """ 将 LCh 转换为 LAB 空间 """
    L, C, h = lch[..., 0], lch[..., 1], lch[..., 2]
    a = C * np.cos(np.radians(h))
    b = C * np.sin(np.radians(h))
    return np.stack([L, a, b], axis=-1)


# 2026.2.5 工业视觉专用版：线性保真映射 (Linear Fidelity Mapping)
# 2026.2.5 增强版：线性增益 + Gamma 调整 默认：0.9/1.2/1， 最佳：0.9/1.3/1.1
def convert_lab_to_srgb(lab_image, linear_gain=0.90, highlight_contrast=1.3, saturation_gain=1.1):
    """
    :param linear_gain: 整体亮度缩放系数。
           <1.0: 压暗整体，给高光留余量，防止过曝死白；
           =1.0: 保持原亮度；
           >1.0: 提亮整体，画面变白，但高光容易死白。
    :param highlight_contrast: Gamma 曲线系数，控制整体明暗和对比度。
           数值越大：整体越暗，暗部压缩越多，对比度越低；
           数值越小：整体越亮，暗部被拉起，对比度越高；
           =1.0: 保持 linear_gain 后的线性亮度关系。
    :param saturation_gain: 饱和度增益，1.0 表示不调整。
    """
    lab_image = lab_image.astype(np.float64)

    # 1. 饱和度调整
    lch = lab_to_lch(lab_image)
    lch[..., 1] *= saturation_gain
    lab_processed = lch_to_lab(lch)

    # 2. 转线性 RGB
    xyz = lab_to_xyz(lab_processed)
    linear_rgb = xyz_to_linear_rgb(xyz)

    # 3. 核心步骤：先缩放亮度，再做 Gamma 调整

    # 3.1 线性亮度缩放 (Headroom)
    # linear_gain < 1 时先压暗，给后续 Gamma 调整留出高光余量，防止死白。
    linear_rgb *= linear_gain
    linear_rgb = np.clip(linear_rgb, 0.0, 1.0)

    # 3.2 Gamma 调整
    # 公式: y = x ^ gamma
    # 作用:
    #   gamma > 1: 压暗暗部，整体变暗，对比度降低；
    #   gamma < 1: 提亮暗部，整体变亮，对比度增加；
    #   gamma = 1: 保持 linear_gain 后的线性关系。
    linear_rgb = np.power(linear_rgb, highlight_contrast)

    # 4. 转 sRGB
    srgb = linear_rgb_to_srgb(linear_rgb)
    srgb_uint8 = (srgb * 255 + 0.5).astype(np.uint8)

    return srgb_uint8


def create_lut(load_model_root_, save_lut_root_, bit_num=8,
               linear_gain=0.90, highlight_contrast=1.3, saturation_gain=1.1):
    mlp_regressor, scaler_X, scaler_L, scaler_a, scaler_b = load_model(load_model_root_)
    generate_rgb_to_lab_lut(mlp_regressor, scaler_X, scaler_L, scaler_a, scaler_b, save_lut_root_, bit_num=bit_num,
                            linear_gain=linear_gain, highlight_contrast=highlight_contrast, saturation_gain=saturation_gain)


def train_data(excel_file_path_, save_root_, train_data_key_dict, sheet_name='sheet1'):
    rgb_input_array, lab_output_array = load_data(excel_file_path_, train_data_key_dict, sheet_name=sheet_name)
    mlp_regressor, scaler_X, scaler_L, scaler_a, scaler_b = trainer(rgb_input_array, lab_output_array)
    save_model(save_root_, mlp_regressor, scaler_X, scaler_L, scaler_a, scaler_b)


# ================= LUT 应用相关 =================

class FileType(Enum):
    IMAGE_TYPE = 1
    OTHER_TYPE = 2


def get_file_path_list(file_root, file_type: FileType):
    if not os.path.exists(file_root):
        print(file_root, "does not exist.")
        return []
    if file_type == FileType.IMAGE_TYPE:
        format_list = ['.jpg', '.jpeg', '.JPG', '.JPEG',
                       '.png', '.tiff', '.tif', '.bmp', '.BMP']
    elif file_type == FileType.OTHER_TYPE:
        format_list = [".json", ".txt", ".xls", ".xlsx"]
    else:
        return []
    path_list = []
    for file_and_dir in os.listdir(file_root):
        if os.path.isfile(os.path.join(file_root, file_and_dir)):
            if os.path.splitext(file_and_dir)[1] in format_list:
                path_list.append(os.path.abspath(os.path.join(file_root, file_and_dir)))
    return path_list


def apply_lut_to_image(image_bgr, lookup_table, gray_scale, quant_step):
    """
    内部原子函数：仅处理单张图像的数据，不涉及IO
    """
    # 1. 输入图是 BGR，必须转为 RGB 才能去查表（因为查表的索引是 R,G,B）
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    height, width = image_rgb.shape[:2]
    rgb_flat = image_rgb.reshape(-1, 3)

    # 2. 量化索引
    indices = (rgb_flat // quant_step).astype(np.int32)
    indices = np.clip(indices, 0, gray_scale - 1)

    # 3. 查表
    # 【注意】因为你修改了生成逻辑，现在 lookup_table 里存的是 RGB 数据！
    # 所以这里查出来的是 RGB
    rgb_result_flat = lookup_table[indices[:, 0], indices[:, 1], indices[:, 2]]

    # 4. 重塑回图像
    result_image_rgb = rgb_result_flat.reshape((height, width, 3))

    # 5. 【新增步骤】格式转换 RGB -> BGR
    # 既然查出来是 RGB，但 OpenCV 保存图片需要 BGR，所以必须转一下
    # 方法1：使用 cvtColor (推荐，兼容性好)
    # result_image_bgr = cv2.cvtColor(result_image_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)

    # 方法2：切片翻转 (速度快)
    result_image_bgr = result_image_rgb[:, :, ::-1]

    return result_image_bgr



def output_rgb2srgb(load_lut_root_, input_path, save_dir=None, bit_num=8):
    """
    通用推理函数：支持单张图片返回，或文件夹批量处理保存
    :param load_lut_root_: LUT文件所在目录
    :param input_path: 可以是单张图片的路径，也可以是包含多张图片的文件夹路径
    :param save_dir: 如果 input_path 是文件夹，则必须提供保存路径
    :param bit_num: 位深
    :return: 如果是单张模式，返回处理后的图像；如果是批量模式，返回 None
    """
    # 1. 预加载 LUT (只加载一次，提高批量处理效率)
    lookup_table_file = os.path.join(load_lut_root_, "rgb_to_srgb.bin")
    if not os.path.exists(lookup_table_file):
        raise FileNotFoundError(f"找不到LUT文件: {lookup_table_file}，请先执行训练。")

    lookup_table = load_lookup_table_bin(lookup_table_file, dtype=np.uint8, bit_num=bit_num)

    gray_scale = pow(2, bit_num)
    quant_step = 256 // gray_scale

    # 2. 判断输入模式
    if os.path.isdir(input_path):
        # =========== 批量模式 ===========
        if save_dir is None:
            raise ValueError("批量处理模式下，必须提供 save_dir 参数")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 获取文件夹下所有图片
        file_list = get_file_path_list(input_path, FileType.IMAGE_TYPE)
        print(f"检测到输入文件夹，共 {len(file_list)} 张图片，开始批量推理...")

        count = 0
        for img_path in file_list:
            # 读取
            img_bgr = imread_cn(img_path)
            if img_bgr is None:
                continue

            # 推理
            res_bgr = apply_lut_to_image(img_bgr, lookup_table, gray_scale, quant_step)

            # 保存 (保留原文件名，增加后缀)
            file_name = os.path.basename(img_path)
            name, ext = os.path.splitext(file_name)
            save_name = f"{name}_calib{ext}"
            save_full_path = os.path.join(save_dir, save_name)

            # 保存图片 (支持中文路径)
            try:
                cv2.imencode(ext if ext else '.jpg', res_bgr)[1].tofile(save_full_path)
                print(f"已保存: {save_name}")
                count += 1
            except Exception as e:
                print(f"保存失败 {save_name}: {e}")

        print(f"批量处理完成，共处理 {count} 张图片。")
        return None

    else:
        # =========== 单张模式 (兼容性保留) ===========
        img_bgr = imread_cn(input_path)
        if img_bgr is None:
            raise FileNotFoundError(f"图片读取失败: {input_path}")

        res_bgr = apply_lut_to_image(img_bgr, lookup_table, gray_scale, quant_step)
        return res_bgr