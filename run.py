import os
import sys
import time
import copy
import logging
import datetime
import configparser
import traceback
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.feature import peak_local_max

"""
工程封装模式：
方法1：pyinstaller run.py --add-data "tools;tools" --hidden-import tools --clean -F --name "ColorCalibrationTool"
说明：这个方法在dist文件夹下就一个.exe程序，但是启动非常慢，优点是非常整洁。

方法2（推荐）：pyinstaller run.py --add-data "tools;tools" --hidden-import tools --clean -D --name "ColorCalibrationTool"
说明：这个方法会在dist路径下生成一个_internal文件夹，里面包含了当前工程的所有依赖打包，不能删除。但是运行启动非常快。
"""

# 假设 tools 包在同级目录或已正确打包
try:
    from tools.calib_utils import train_data, create_lut, output_rgb2srgb
except ImportError:
    # 开发环境下如果没有安装 tools 包可能会报错，这里忽略，运行时需确保环境正确
    pass

# ======================== 路径与环境配置 ========================
if getattr(sys, 'frozen', False):
    # 打包后的运行路径 (Temp资源目录)
    BASE_DIR = sys._MEIPASS
    # 用户实际存放 EXE 的目录 (配置文件、日志、图片都在这里)
    EXE_OUTER_DIR = os.path.dirname(sys.executable)
else:
    # 正常 Python 开发环境路径
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    EXE_OUTER_DIR = BASE_DIR


# ======================== 1. 中文路径支持工具 ========================
def cv_imread_utf8(file_path):
    """
    支持中文路径的图片读取
    """
    try:
        return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    except Exception as e:
        logging.error(f"读取图片失败: {file_path}, 错误: {e}")
        return None


def cv_imwrite_utf8(file_path, img):
    """
    支持中文路径的图片保存
    """
    try:
        suffix = os.path.splitext(file_path)[1]
        # 如果没有后缀，默认用bmp
        if not suffix:
            suffix = '.bmp'
        cv2.imencode(suffix, img)[1].tofile(file_path)
        return True
    except Exception as e:
        logging.error(f"保存图片失败: {file_path}, 错误: {e}")
        return False


# ======================== 2. 日志系统初始化 ========================
def setup_logging():
    log_dir = os.path.join(EXE_OUTER_DIR, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(log_dir, f"{date_str}.log")

    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),  # 写入文件
            logging.StreamHandler(sys.stdout)  # 输出到控制台
        ],
        force=True  # 强制重新配置，防止被其他库抢占 (Python 3.8+)
    )
    logging.info("=" * 30)
    logging.info(f"程序启动。日志文件路径: {log_file}")
    logging.info(f"执行环境: {'Frozen EXE' if getattr(sys, 'frozen', False) else 'Python Script'}")


# ======================== 3. 配置文件管理 (.ini) ========================
class ConfigManager:
    def __init__(self):
        self.config_path = os.path.join(EXE_OUTER_DIR, "config.ini")
        self.cfg = configparser.ConfigParser()
        self.load_or_create()

    def load_or_create(self):
        if not os.path.exists(self.config_path):
            logging.info("未检测到配置文件，正在生成默认 config.ini ...")
            self.cfg['PATHS'] = {
                'templatedir': 'data/1_template',
                'capturedir': 'data/2_capture',
                'resultdir': 'data/3_result',
                'testdir': 'data/4_test_data'
            }
            self.cfg['MATCHING'] = {
                'scale': '0.5',
                'marginratio': '0.25',
                'matchscore': '0.6'
            }
            self.cfg['GRID'] = {
                'rownum': '49',
                'colnum': '33'
            }
            self.cfg['DEBUG'] = {
                'patchradius': '30',
                'showdebugimages': 'True'
            }
            self.cfg['COLOR'] = {
                'linear_gain': '0.90',
                'highlight_contrast': '1.3',
                'saturation_gain': '1.1'
            }
            with open(self.config_path, 'w', encoding='utf-8-sig') as f:
                self.cfg.write(f)
        else:
            logging.info(f"加载配置文件: {self.config_path}")
            self.cfg.read(self.config_path, encoding='utf-8')

    def get_path(self, key):
        folder_name = self.cfg.get('PATHS', key, fallback='')
        return os.path.join(EXE_OUTER_DIR, folder_name)

    def get_float(self, section, key, default=0.0):
        return self.cfg.getfloat(section, key, fallback=default)

    def get_int(self, section, key, default=0):
        return self.cfg.getint(section, key, fallback=default)


# 全局配置对象占位符，在 main 中初始化
config_manager = None


# ======================== 核心业务逻辑 ========================

def list_images(path):
    IMAGE_SUFFIX = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG',
                    '.tiff', '.TIFF', '.bmp', '.BMP', '.tif', '.TIF']
    path_list = []
    if not os.path.exists(path):
        logging.warning(f"文件夹不存在: {path}")
        return []

    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if os.path.splitext(filename)[1] in IMAGE_SUFFIX:
                path_list.append(os.path.join(dirpath, filename))
    return path_list


def match_coarse(resized_img, resized_template):
    res = cv2.matchTemplate(resized_img, resized_template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    return max_loc, max_val


def match_fine(original_img, original_template, coarse_x, coarse_y, scale, margin_ratio):
    h_t, w_t = original_template.shape[:2]
    cx = int(coarse_x / scale)
    cy = int(coarse_y / scale)

    margin_x = int(w_t * margin_ratio)
    margin_y = int(h_t * margin_ratio)
    x1 = max(0, cx - margin_x)
    y1 = max(0, cy - margin_y)
    x2 = min(original_img.shape[1], cx + margin_x + w_t)
    y2 = min(original_img.shape[0], cy + margin_y + h_t)

    # 增加边界检查，防止切图越界导致程序崩溃
    if y2 <= y1 or x2 <= x1:
        return 0, 0, 0

    search_region = original_img[y1:y2, x1:x2]

    # 增加对搜索区域尺寸的校验
    if search_region.shape[0] < h_t or search_region.shape[1] < w_t:
        return 0, 0, 0

    if search_region.size == 0:
        return 0, 0, 0

    res = cv2.matchTemplate(search_region, original_template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    fine_x = x1 + max_loc[0]
    fine_y = y1 + max_loc[1]
    return fine_x, fine_y, max_val


def match_lut_roi(temp_img, match_image):
    # 使用全局 config_manager
    scale = config_manager.get_float('MATCHING', 'scale', 0.5)
    margin_ratio = config_manager.get_float('MATCHING', 'marginratio', 0.25)
    match_score_thr = config_manager.get_float('MATCHING', 'matchscore', 0.6)

    resize_temp_img = cv2.resize(temp_img, None, fx=scale, fy=scale)
    resize_match_image = cv2.resize(match_image, None, fx=scale, fy=scale)

    # 粗匹配
    coarse_loc, coarse_score = match_coarse(resize_match_image, resize_temp_img)

    # 精匹配
    start_x, start_y, fine_score = match_fine(match_image, temp_img, coarse_loc[0], coarse_loc[1], scale, margin_ratio)

    logging.info(f"模板匹配结果 -> 得分: {fine_score:.4f} (阈值: {match_score_thr}), 位置: ({start_x}, {start_y})")

    if fine_score >= match_score_thr:
        crop_h, crop_w = temp_img.shape[0], temp_img.shape[1]
        return match_image[start_y:start_y + crop_h, start_x:start_x + crop_w, :].copy()
    else:
        logging.warning(f"匹配得分 {fine_score:.4f} 低于阈值 {match_score_thr}，匹配失败。")
        return None


def validate_grid_robust(centers, img_shape):
    rows = config_manager.get_int('GRID', 'rownum', 49)
    cols = config_manager.get_int('GRID', 'colnum', 33)

    pts = np.array(centers)
    h_img, w_img = img_shape[:2]

    if len(centers) == 0:
        return []

    dx, dy = w_img / cols, h_img / rows
    yy, xx = np.mgrid[0:rows, 0:cols]
    theoretical_grid = np.vstack([((xx + 0.5) * dx).ravel(), ((yy + 0.5) * dy).ravel()]).T

    diffs = []
    for t_pt in theoretical_grid:
        if len(pts) > 0:
            dists = np.linalg.norm(pts - t_pt, axis=1)
            nearest_idx = np.argmin(dists)
            if dists[nearest_idx] < dx * 0.4:
                diffs.append(pts[nearest_idx] - t_pt)

    offset = np.median(diffs, axis=0) if diffs else np.array([0, 0])
    corrected_grid = theoretical_grid + offset

    final_centers = []
    for c_pt in corrected_grid:
        if len(pts) > 0:
            dists = np.linalg.norm(pts - c_pt, axis=1)
            min_idx = np.argmin(dists)
            if dists[min_idx] < 20:
                final_centers.append(list(pts[min_idx].astype(int)))
            else:
                final_centers.append(list(c_pt.astype(int)))
        else:
            final_centers.append(list(c_pt.astype(int)))

    return final_centers


def extract_all_mosaic_centers(img):
    if img is None:
        logging.error("传入 extract_all_mosaic_centers 的图像为空")
        return []

    try:
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        l_channel = hls[:, :, 1]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l_channel)
        blurred = cv2.GaussianBlur(enhanced_l, (9, 9), 0)
        edges = cv2.Canny(blurred, 30, 80)

        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        dist_mask = cv2.bitwise_not(edges)
        dist_transform = cv2.distanceTransform(dist_mask, cv2.DIST_L2, 5)

        coordinates = peak_local_max(dist_transform, min_distance=50, threshold_abs=5.0)

        centers = []
        for coord in coordinates:
            py, px = int(coord[0]), int(coord[1])
            centers.append([px, py])

        final_centers = validate_grid_robust(centers, img.shape)
        logging.info(f"网格提取 -> 原始点数: {len(centers)}, 修正后点数: {len(final_centers)}")
        return final_centers
    except Exception as e:
        logging.error(f"提取网格中心点时出错: {e}", exc_info=True)
        return []


def save_debug_image_with_indices(img, centers, save_name):
    if img is None or not centers:
        return

    save_result_root = config_manager.get_path('resultdir')
    vis_img = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for idx, pt in enumerate(centers):
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(vis_img, (x, y), 3, (0, 0, 255), -1)
        cv2.putText(vis_img, str(idx), (x - 10, y - 5), font, 0.4, (0, 255, 0), 1)

    save_path = os.path.join(save_result_root, save_name)
    success = cv_imwrite_utf8(save_path, vis_img)
    if success:
        logging.info(f"已保存调试图片: {save_name}")


# ======================== 功能模块 1：训练与标定 ========================

def train_and_generate_lut():
    # 1. 获取配置
    input_temp_path = config_manager.get_path('templatedir')
    input_insp_path = config_manager.get_path('capturedir')
    save_result_root = config_manager.get_path('resultdir')
    patch_radius = config_manager.get_int('DEBUG', 'patchradius', 30)

    # 2. 创建目录
    for folder in [input_temp_path, input_insp_path, save_result_root]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            logging.info(f"创建文件夹: {folder}")

    # 3. 检查文件
    temp_imgs = list_images(input_temp_path)
    insp_imgs = list_images(input_insp_path)

    if not temp_imgs:
        logging.error(f"模板文件夹 {input_temp_path} 为空！")
        print(f"错误：模板文件夹 {input_temp_path} 中没有图片。")
        return
    if not insp_imgs:
        logging.error(f"采集文件夹 {input_insp_path} 为空！")
        print(f"错误：采集文件夹 {input_insp_path} 中没有图片。")
        return

    steps = [
        "step1: 加载本地色卡ROI小图",
        "step2: 加载采集图片大图并进行匹配",
        "step3: 计算Patch RGB均值",
        "step4: 数据转换 BGR->RGB",
        "step5: 合并数据到 Excel",
        "step6: 训练颜色映射数据",
        "step7: 创建查找表 (LUT)"
    ]

    print("\n开始执行训练与生成 LUT 流程...")
    logging.info(">>> 开始执行训练与生成 LUT 流程 (共7步)")

    try:
        with tqdm(total=len(steps), desc=">>> 训练进度", bar_format='{l_bar}{bar:30}{r_bar}') as pbar:

            # --- Step 1 ---
            pbar.set_description(steps[0])
            logging.info(f"[Step 1] 开始: 加载模板图 {temp_imgs[0]}")
            temp_img = cv_imread_utf8(temp_imgs[0])
            if temp_img is None:
                raise Exception(f"模板图读取失败: {temp_imgs[0]}")

            # 旋转90度逆时针
            temp_img_rotate = cv2.rotate(temp_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            center_pdf_list = extract_all_mosaic_centers(temp_img_rotate)
            save_debug_image_with_indices(temp_img_rotate, center_pdf_list, "1_test_temp_location.jpg")
            pbar.update(1)

            # --- Step 2 ---
            pbar.set_description(steps[1])
            logging.info(f"[Step 2] 开始: 加载采集图 {insp_imgs[0]} 并进行匹配")
            match_image = cv_imread_utf8(insp_imgs[0])
            if match_image is None:
                raise Exception(f"采集图读取失败: {insp_imgs[0]}")

            match_img = match_lut_roi(temp_img, match_image)
            if match_img is None:
                raise Exception("ROI 匹配失败，请检查图片质量或调整 config.ini 中的匹配阈值")

            match_img_rotate = cv2.rotate(match_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            center_insp_list = extract_all_mosaic_centers(match_img_rotate)
            save_debug_image_with_indices(match_img_rotate, center_insp_list, "2_test_insp_location.jpg")
            pbar.update(1)

            # --- Step 3 ---
            pbar.set_description(steps[2])
            logging.info("[Step 3] 开始: 提取色块均值")
            if len(center_insp_list) == 0:
                raise Exception("未检测到任何色块中心点，无法计算均值")

            bgr_insp_list = []
            for pt in center_insp_list:
                y_min = max(0, pt[1] - patch_radius)
                y_max = min(match_img_rotate.shape[0], pt[1] + patch_radius)
                x_min = max(0, pt[0] - patch_radius)
                x_max = min(match_img_rotate.shape[1], pt[0] + patch_radius)

                roi = match_img_rotate[y_min:y_max, x_min:x_max, :]
                bgr_insp_list.append(np.mean(roi, axis=(0, 1)))

            logging.info(f"提取了 {len(bgr_insp_list)} 个色块的均值")
            pbar.update(1)

            # --- Step 4 ---
            pbar.set_description(steps[3])
            logging.info("[Step 4] 开始: 格式转换 BGR -> RGB")
            rows = []
            for bgr in bgr_insp_list:
                rows.append([bgr[2], bgr[1], bgr[0]])
            pbar.update(1)

            # --- Step 5 ---
            pbar.set_description(steps[4])
            logging.info("[Step 5] 开始: 生成/合并 Excel 数据")
            df_new_rgb = pd.DataFrame(rows, columns=['R', 'G', 'B'])
            map_file_path = os.path.join(input_temp_path, "map.xlsx")

            # 必须确保 map.xlsx 存在，否则无法训练
            if not os.path.exists(map_file_path):
                raise FileNotFoundError(f"严重错误：在 {input_temp_path} 下找不到 map.xlsx！\n"
                                        "该文件必须包含标准色卡的 LAB_L, LAB_A, LAB_B 数值。")

            try:
                df_map = pd.read_excel(map_file_path)
                # 检查是否包含必要的列
                required_cols = ['LAB_L', 'LAB_A', 'LAB_B']
                if not all(col in df_map.columns for col in required_cols):
                    raise ValueError(f"map.xlsx 缺少必要的标准值列: {required_cols}")

                if len(df_map) != len(df_new_rgb):
                    logging.warning(f"行数不匹配警告: Excel标准值({len(df_map)}) vs 识别采集值({len(df_new_rgb)})")

                min_len = min(len(df_map), len(df_new_rgb))
                # 更新 R, G, B 列
                df_map.loc[:min_len - 1, 'R'] = df_new_rgb.loc[:min_len - 1, 'R']
                df_map.loc[:min_len - 1, 'G'] = df_new_rgb.loc[:min_len - 1, 'G']
                df_map.loc[:min_len - 1, 'B'] = df_new_rgb.loc[:min_len - 1, 'B']
                final_df = df_map
            except Exception as e:
                logging.error(f"处理 map.xlsx 失败: {e}", exc_info=True)
                raise e

            excel_save_path = os.path.join(save_result_root, 'map_data_result.xlsx')
            final_df.to_excel(excel_save_path, index=False)
            logging.info(f"Excel 已保存至: {excel_save_path}")
            pbar.update(1)

            # --- Step 6 ---
            pbar.set_description(steps[5])
            logging.info("[Step 6] 开始: 训练颜色映射模型 (MLP)")
            train_data_key_dict = {
                'input1': 'R', 'input2': 'G', 'input3': 'B',
                'output1': 'LAB_L', 'output2': 'LAB_A', 'output3': 'LAB_B'
            }
            train_data(excel_save_path, save_result_root, train_data_key_dict, sheet_name='Sheet1')
            logging.info("模型训练完成并保存")
            pbar.update(1)

            # --- Step 7 ---
            pbar.set_description(steps[6])
            logging.info("[Step 7] 开始: 创建查找表 (LUT)")
            linear_gain = config_manager.get_float('COLOR', 'linear_gain', 0.90)
            highlight_contrast = config_manager.get_float('COLOR', 'highlight_contrast', 1.3)
            saturation_gain = config_manager.get_float('COLOR', 'saturation_gain', 1.1)
            logging.info(f"LUT颜色参数 -> linear_gain={linear_gain}, highlight_contrast={highlight_contrast}, saturation_gain={saturation_gain}")
            create_lut(save_result_root, save_result_root, bit_num=8,
                       linear_gain=linear_gain, highlight_contrast=highlight_contrast, saturation_gain=saturation_gain)
            logging.info("LUT 创建完成")
            pbar.update(1)

        print("\n>>> 训练成功！LUT文件已生成在 data/3_result 文件夹。")
        logging.info(">>> 训练流程成功结束")

    except Exception as e:
        logging.error("！！！ 训练流程发生严重错误 ！！！", exc_info=True)
        print(f"\n！！！ 发生错误: {e}")
        print("详细错误已保存到 logs 文件夹下的日志中。")


# ======================== 功能模块 2：批量推理 ========================

def run_inference_test():
    """
    独立的功能模块：加载标定结果，对 data/4_test_data/input 中的图片进行矫正，结果存入 data/4_test_data/output
    """
    save_result_root = config_manager.get_path('resultdir')
    test_root = config_manager.get_path('testdir')

    # 构造 input 和 output 路径
    input_dir = os.path.join(test_root, 'input')
    output_dir = os.path.join(test_root, 'output')

    print("\n开始执行批量推理测试...")
    logging.info(">>> 开始执行批量推理测试")

    # 1. 检查 LUT 是否存在
    lut_path = os.path.join(save_result_root, "rgb_to_srgb.bin")
    if not os.path.exists(lut_path):
        print(f"错误：找不到标定文件 {lut_path}")
        print("请先运行功能 1 进行训练和生成 LUT。")
        logging.error("推理失败：LUT文件不存在")
        return

    # 2. 检查输入文件夹
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
        print(f"提示：输入文件夹 {input_dir} 不存在，已自动创建。")
        print("请将待矫正的图片放入该文件夹，然后重新运行此功能。")
        logging.warning(f"输入文件夹为空或不存在: {input_dir}")
        return

    # 3. 检查是否有图片
    test_images = list_images(input_dir)
    if len(test_images) == 0:
        print(f"警告：输入文件夹 {input_dir} 为空。")
        print("请放入待测试图片。")
        logging.warning("输入文件夹中没有检测到图片")
        return

    # 4. 执行推理
    try:
        logging.info(f"检测到 {len(test_images)} 张图片，结果将保存至 {output_dir}")
        output_rgb2srgb(
            load_lut_root_=save_result_root,
            input_path=input_dir,  # 输入 input 文件夹
            save_dir=output_dir,  # 输出 output 文件夹
            bit_num=8
        )
        print(f"\n>>> 推理完成！结果已保存在: {output_dir}")
        logging.info("批量推理流程成功结束")

    except Exception as e:
        logging.error("推理流程发生错误", exc_info=True)
        print(f"\n！！！ 推理发生错误: {e}")


# ======================== 主程序入口 ========================
if __name__ == '__main__':
    # 1. 【先】启动日志
    setup_logging()

    # 2. 【后】初始化全局配置 (直接赋值，无需 global)
    config_manager = ConfigManager()

    while True:
        print("\n" + "=" * 56)
        print("***************** 相机色彩矫正标定系统 *****************")
        print(f"当前工作目录: {EXE_OUTER_DIR}")
        print("-> 功能 1： 训练模型并生成标定文件 (Train & LUT)")
        print("-> 功能 2： 批量推理测试图片 (Inference)")
        print("-> 功能 3： 重新加载配置文件 (Reload Config)")
        print("-> 其他  ： 退出程序")
        print("=" * 56)

        try:
            user_input = input("请选择功能号码: ").strip()

            if user_input == '1':
                if config_manager is None: config_manager = ConfigManager()
                train_and_generate_lut()

            elif user_input == '2':
                if config_manager is None: config_manager = ConfigManager()
                run_inference_test()

            elif user_input == '3':
                if config_manager is None: config_manager = ConfigManager()
                config_manager.load_or_create()
                print(">>> 配置文件已重新加载。")

            else:
                logging.info("用户主动退出程序")
                break

        except KeyboardInterrupt:
            logging.info("用户强制中断 (Ctrl+C)")
            break
        except Exception as e:
            # 捕获主循环的未知异常，防止直接闪退
            logging.critical("主循环发生未捕获异常", exc_info=True)
            print(f"\n系统严重错误: {e}")
            print("请将 logs 文件夹发给开发人员进行排查。")
            time.sleep(1)

    input("按回车键关闭窗口...")