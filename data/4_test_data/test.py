"""色标条端点色块宽度检测示例脚本。

本脚本读取当前目录下的 test.bmp，判断起点和终点所在色块宽度
是否与其邻域标准色块一致，并输出中间步骤的可视化曲线。
"""

import os
import sys
import io
import numpy as np

# 在某些 Windows 终端下默认编码为 GBK，强制使用 UTF-8 输出中文
if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    except Exception:
        pass

# 将项目根目录加入模块搜索路径，以便导入 tools 包
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib
import matplotlib.font_manager as fm

matplotlib.use("Agg")  # 不使用 GUI 后端，适合后台保存图片

# 尝试设置中文字体，优先使用系统黑体/雅黑，避免中文标题显示为方框
_CANDIDATE_FONTS = [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
]
_font_set = False
for _fp in _CANDIDATE_FONTS:
    if os.path.exists(_fp):
        try:
            matplotlib.rcParams["font.sans-serif"] = [fm.FontProperties(fname=_fp).get_name()]
            matplotlib.rcParams["axes.unicode_minus"] = False
            _font_set = True
            break
        except Exception:
            continue
if not _font_set:
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt

from tools.block_detector import (
    analyze_color_bar_endpoints,
    extract_horizontal_boundaries,
    compute_block_widths,
)


def _imread_fallback(path: str) -> np.ndarray:
    """兼容读取图像，优先 cv2，失败时使用 PIL 兜底。

    Args:
        path: 图像文件路径。

    Returns:
        RGB 格式 numpy 数组。

    Raises:
        RuntimeError: 读取失败。
    """
    try:
        import cv2
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("cv2 解码返回空")
        # OpenCV 默认读取 BGR，转换为 RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    except Exception:
        try:
            from PIL import Image
            with Image.open(path) as pil_img:
                return np.array(pil_img.convert("RGB"))
        except Exception as e:
            raise RuntimeError(f"无法读取图像: {path}, 错误: {e}") from e


def _smooth_scan_line(
    image: np.ndarray,
    y: int,
    row_span: int,
) -> np.ndarray:
    """提取并平滑扫描线颜色曲线。"""
    h = image.shape[0]
    y0 = max(0, y - row_span)
    y1 = min(h, y + row_span + 1)
    strip = image[y0:y1, :, :]
    return np.mean(strip, axis=0)


def _compute_gradient(curve: np.ndarray) -> np.ndarray:
    """计算相邻像素 L2 色差梯度。"""
    return np.linalg.norm(curve[1:] - curve[:-1], axis=1)


def _cluster_edges(edges: np.ndarray, cluster_gap: int) -> np.ndarray:
    """将邻近边缘聚类为真实边界。"""
    if edges.size == 0:
        return np.array([], dtype=np.float32)

    boundaries: list[float] = []
    group_start = int(edges[0])
    group_end = group_start

    for x in edges[1:]:
        x_int = int(x)
        if x_int - group_end <= cluster_gap:
            group_end = x_int
        else:
            boundaries.append((group_start + group_end) / 2.0)
            group_start = x_int
            group_end = x_int

    boundaries.append((group_start + group_end) / 2.0)
    return np.array(boundaries, dtype=np.float32)


def visualize_detection_process(
    image: np.ndarray,
    x: int,
    y: int,
    direction: str,
    scan_range_px: int,
    edge_threshold: float,
    row_span: int,
    output_path: str,
) -> None:
    """绘制某一侧（起点右侧或终点左侧）的检测过程曲线。

    Args:
        image: 输入图像。
        x: 待检测点 x 坐标。
        y: 待检测点 y 坐标。
        direction: "right" 或 "left"，决定展示哪一侧。
        scan_range_px: 扫描像素范围。
        edge_threshold: 边缘梯度阈值。
        row_span: 扫描线平滑行数。
        output_path: 输出图片保存路径。
    """
    h, w = image.shape[:2]

    # 确定展示窗口
    if direction == "right":
        x0 = x
        x1 = min(w - 1, x + scan_range_px)
    else:
        x0 = max(0, x - scan_range_px)
        x1 = x

    # 提取并平滑扫描线
    curve_full = _smooth_scan_line(image, y, row_span)
    grad_full = _compute_gradient(curve_full)
    edges_full = np.where(grad_full > edge_threshold)[0]
    boundaries_full = _cluster_edges(edges_full, cluster_gap=3)

    # 截取窗口内数据
    curve = curve_full[x0:x1]
    grad = grad_full[x0:x1]
    xs = np.arange(x0, x1)

    # 窗口内的边界
    boundaries = boundaries_full[(boundaries_full >= x0) & (boundaries_full <= x1)]

    # 找到 x 所在区间
    if x >= boundaries_full[-1]:
        left_b = float(boundaries_full[-1])
        right_b = float(w - 1)
    else:
        idx = int(np.searchsorted(boundaries_full, x)) - 1
        left_b = float(boundaries_full[idx])
        right_b = float(boundaries_full[idx + 1])

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    axes[1].sharex(axes[0])
    fig.suptitle(f"{'起点右侧' if direction == 'right' else '终点左侧'}检测过程 (y={y})", fontsize=14)

    # ---------- 子图 1：亮度曲线 ----------
    ax = axes[0]
    brightness = np.mean(curve, axis=1)
    ax.plot(xs, brightness, color="black", linewidth=1.0, label="平滑后亮度")
    ax.axvline(x, color="red", linestyle="--", linewidth=2, label=f"检测点 x={x}")
    for b in boundaries:
        ax.axvline(b, color="blue", linestyle=":", alpha=0.6)
    ax.axvspan(left_b, right_b, color="green", alpha=0.15, label="当前色块")
    ax.set_ylabel("亮度")
    ax.set_title("步骤 1~2：扫描线亮度（已多行平均）")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # ---------- 子图 2：色差梯度 + 阈值 ----------
    ax = axes[1]
    ax.plot(xs, grad, color="darkorange", linewidth=1.2, label="色差梯度 G[x]")
    ax.axhline(edge_threshold, color="purple", linestyle="--", linewidth=2, label=f"阈值 T={edge_threshold}")
    # 候选边缘
    cand_in_win = edges_full[(edges_full >= x0) & (edges_full <= x1)]
    ax.scatter(cand_in_win, grad_full[cand_in_win], color="red", s=15, zorder=5, label="候选边缘")
    # 聚类边界
    for b in boundaries:
        ax.axvline(b, color="blue", linestyle=":", alpha=0.8, label="聚类边界" if b == boundaries[0] else "")
    ax.set_ylabel("梯度")
    ax.set_title("步骤 3~4：梯度阈值提取 + 边缘聚类")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # ---------- 子图 3：色块宽度分布 ----------
    ax = axes[2]
    widths = compute_block_widths(boundaries_full)
    # 只取窗口内的完整色块宽度
    ref_widths = []
    for i in range(len(boundaries_full) - 1):
        b_left = boundaries_full[i]
        b_right = boundaries_full[i + 1]
        if direction == "right" and b_left >= x0 and b_right <= x1 and b_left >= right_b:
            ref_widths.append(b_right - b_left)
        elif direction == "left" and b_left >= x0 and b_right <= x1 and b_right <= left_b:
            ref_widths.append(b_right - b_left)

    current_width = right_b - left_b
    std_width = float(np.median(ref_widths)) if ref_widths else 0.0

    bar_labels = ["当前色块"] + [f"参考{i+1}" for i in range(len(ref_widths))]
    bar_values = [current_width] + ref_widths
    colors = ["green"] + ["steelblue"] * len(ref_widths)
    bar_positions = np.arange(len(bar_values))
    ax.bar(bar_positions, bar_values, color=colors, alpha=0.7)
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(bar_labels, rotation=15, ha="right")
    if std_width > 0:
        ax.axhline(std_width, color="red", linestyle="--", linewidth=2, label=f"标准宽度={std_width:.1f}")
    ax.set_ylabel("宽度（像素）")
    ax.set_title(f"步骤 5~7：当前色块 vs 参考色块宽度（ratio={current_width/std_width:.3f}）")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    axes[0].set_xlim(x0, x1)
    plt.xlabel("x 坐标（像素）")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"可视化已保存: {output_path}")


def main() -> None:
    """主入口：读取图片并执行端点色块宽度判断与可视化。"""
    input_img_path = os.path.join(os.path.dirname(__file__), "test.bmp")
    resize_ratio = 1
    start_pos_x, start_pos_y = 42 * resize_ratio, 490 * resize_ratio
    end_pos_x, end_pos_y = 13983 * resize_ratio, 490 * resize_ratio
    scan_range_px = 1000
    edge_threshold = 20.0
    row_span = 10

    print(f"读取图片: {input_img_path}")
    image = _imread_fallback(input_img_path)
    print(f"图片尺寸: {image.shape}")

    result = analyze_color_bar_endpoints(
        image=image,
        start_pos=(start_pos_x, start_pos_y),
        end_pos=(end_pos_x, end_pos_y),
        scan_range_px=scan_range_px,
        edge_threshold=edge_threshold,
    )

    print("\n===== 起点检测结果 =====")
    start = result["start"]
    print(f"坐标: ({start['x']}, {start['y']})")
    print(f"当前色块边界: [{start['boundaries'][0]:.1f}, {start['boundaries'][1]:.1f}]")
    print(f"当前色块宽度: {start['block_width']:.1f}")
    print(f"参考标准宽度: {start['std_width']:.1f}")
    print(f"比例: {start['ratio']:.3f}")
    print(f"分类: {start['category']} ({'完整色块' if start['category'] == 'full' else ('半块' if start['category'] == 'half' else '未知')})")
    print(f"参考色块数量: {len(start['reference_widths'])}")

    print("\n===== 终点检测结果 =====")
    end = result["end"]
    print(f"坐标: ({end['x']}, {end['y']})")
    print(f"当前色块边界: [{end['boundaries'][0]:.1f}, {end['boundaries'][1]:.1f}]")
    print(f"当前色块宽度: {end['block_width']:.1f}")
    print(f"参考标准宽度: {end['std_width']:.1f}")
    print(f"比例: {end['ratio']:.3f}")
    print(f"分类: {end['category']} ({'完整色块' if end['category'] == 'full' else ('半块' if end['category'] == 'half' else '未知')})")
    print(f"参考色块数量: {len(end['reference_widths'])}")

    # 输出可视化
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    visualize_detection_process(
        image=image,
        x=start_pos_x,
        y=start_pos_y,
        direction="right",
        scan_range_px=scan_range_px,
        edge_threshold=edge_threshold,
        row_span=row_span,
        output_path=os.path.join(output_dir, "start_right_process.png"),
    )

    visualize_detection_process(
        image=image,
        x=end_pos_x,
        y=end_pos_y,
        direction="left",
        scan_range_px=scan_range_px,
        edge_threshold=edge_threshold,
        row_span=row_span,
        output_path=os.path.join(output_dir, "end_left_process.png"),
    )


if __name__ == "__main__":
    main()
