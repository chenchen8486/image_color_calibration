"""色标条色块宽度检测模块。

用于判断给定坐标所在色块宽度与邻域标准色块宽度是否一致，
典型场景为判断起点/终点色块是否为完整色块或半块。
"""

from typing import List, Tuple, Literal, Optional
import numpy as np


# 默认参数配置，便于统一调参
DEFAULT_EDGE_THRESHOLD: float = 20.0
DEFAULT_ROW_SPAN: int = 10
DEFAULT_CLUSTER_GAP: int = 3
DEFAULT_SCAN_RANGE_PX: int = 1000
DEFAULT_FULL_RATIO_RANGE: Tuple[float, float] = (0.75, 1.25)
DEFAULT_HALF_RATIO_RANGE: Tuple[float, float] = (0.35, 0.65)


def _validate_image(image: np.ndarray) -> np.ndarray:
    """校验并规范化图像数组。

    Args:
        image: 输入图像，支持 (H, W) 或 (H, W, C)。

    Returns:
        二维或三维 float32 数组。

    Raises:
        ValueError: 图像维度或类型不合法。
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("输入 image 必须是 numpy.ndarray")
    if image.ndim not in (2, 3):
        raise ValueError(f"图像维度必须是 2 或 3，当前为 {image.ndim}")
    return image.astype(np.float32)


def _smooth_scan_line(
    image: np.ndarray,
    y: int,
    row_span: int,
) -> np.ndarray:
    """提取并平滑扫描线颜色曲线。

    取 y 附近 [y-row_span, y+row_span] 共 2*row_span+1 行，
    沿 Y 方向做均值，以降低单行噪声。

    Args:
        image: 输入图像，(H, W, C) 或 (H, W)。
        y: 扫描线中心行。
        row_span: 扫描线上下扩展行数。

    Returns:
        形状为 (W, C) 或 (W,) 的 float32 曲线。
    """
    h = image.shape[0]
    y0 = max(0, y - row_span)
    y1 = min(h, y + row_span + 1)
    strip = image[y0:y1, :, :] if image.ndim == 3 else image[y0:y1, :]
    return np.mean(strip, axis=0)


def _cluster_edges(edges: np.ndarray, cluster_gap: int) -> np.ndarray:
    """将邻近边缘聚类为真实边界。

    抗锯齿/渐变边界通常会在 2~3 个连续像素上产生跳变，
    本函数将间隔不超过 cluster_gap 的边缘归为一组，取均值作为边界。

    Args:
        edges: 候选边缘 x 坐标数组。
        cluster_gap: 同组边缘最大像素间隔。

    Returns:
        聚类后的边界位置数组。
    """
    if edges.size == 0:
        return np.array([], dtype=np.float32)

    boundaries: List[float] = []
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


def extract_horizontal_boundaries(
    image: np.ndarray,
    y: int,
    row_span: int = DEFAULT_ROW_SPAN,
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
    cluster_gap: int = DEFAULT_CLUSTER_GAP,
) -> np.ndarray:
    """提取水平方向色块边界。

    Args:
        image: 输入图像，(H, W, C) 或 (H, W)。
        y: 扫描线中心 y 坐标。
        row_span: 扫描线上下扩展行数，用于降噪。
        edge_threshold: 相邻像素色差阈值，超过视为候选边缘。
        cluster_gap: 边缘聚类最大像素间隔。

    Returns:
        边界位置数组，形状 (M,)，float32。

    Raises:
        ValueError: 输入参数不合法。
    """
    image = _validate_image(image)
    if y < 0 or y >= image.shape[0]:
        raise ValueError(f"y 坐标越界: {y}，图像高度: {image.shape[0]}")
    if edge_threshold <= 0:
        raise ValueError("edge_threshold 必须大于 0")
    if cluster_gap < 0:
        raise ValueError("cluster_gap 必须大于等于 0")

    curve = _smooth_scan_line(image, y, row_span)
    # 计算相邻像素 L2 色差梯度
    grad = np.linalg.norm(curve[1:] - curve[:-1], axis=1)
    edges = np.where(grad > edge_threshold)[0]
    return _cluster_edges(edges, cluster_gap)


def compute_block_widths(boundaries: np.ndarray) -> np.ndarray:
    """计算相邻边界间距，即色块宽度。

    Args:
        boundaries: 已排序的边界位置数组。

    Returns:
        色块宽度数组。
    """
    if boundaries.size < 2:
        return np.array([], dtype=np.float32)
    return np.diff(boundaries)


def _find_containing_segment(
    boundaries: np.ndarray,
    x: int,
    image_width: int,
) -> Optional[Tuple[int, float, float]]:
    """找到 x 所在的色块区间。

    支持坐标落在第一个边界之前或最后一个边界之后的情况，
    此时把图像边缘（0 或 width-1）作为缺失边界。

    Args:
        boundaries: 已排序的边界位置数组。
        x: 待查询 x 坐标。
        image_width: 图像宽度。

    Returns:
        (索引 i, 左边界 b_i, 右边界 b_{i+1})，若 x 不在任何有效区间内则返回 None。
    """
    if boundaries.size == 0:
        return None

    first = float(boundaries[0])
    last = float(boundaries[-1])

    # 位于第一个边界左侧，将 0 作为左边界
    if x <= first:
        if 0 <= x <= first:
            return -1, 0.0, first
        return None

    # 位于最后一个边界右侧，将图像右边缘作为右边界
    if x >= last:
        if last <= x < image_width:
            return boundaries.size - 1, last, float(image_width - 1)
        return None

    idx = int(np.searchsorted(boundaries, x)) - 1
    if 0 <= idx < boundaries.size - 1:
        left = float(boundaries[idx])
        right = float(boundaries[idx + 1])
        if left <= x <= right:
            return idx, left, right

    return None


def classify_endpoint_block(
    image: np.ndarray,
    x: int,
    y: int,
    direction: Literal["left", "right"],
    scan_range_px: int = DEFAULT_SCAN_RANGE_PX,
    row_span: int = DEFAULT_ROW_SPAN,
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
    cluster_gap: int = DEFAULT_CLUSTER_GAP,
    full_ratio_range: Tuple[float, float] = DEFAULT_FULL_RATIO_RANGE,
    half_ratio_range: Tuple[float, float] = DEFAULT_HALF_RATIO_RANGE,
) -> dict:
    """判断指定坐标所在色块宽度与邻域标准色块是否一致。

    从 x 开始向指定方向扫描 scan_range_px 像素，收集该范围内的参考色块宽度，
    以中位数作为标准宽度，再与 x 所在色块宽度做比例比较。

    Args:
        image: 输入图像，(H, W, C) 或 (H, W)。
        x: 待判断点 x 坐标。
        y: 待判断点 y 坐标。
        direction: 扫描方向，"left" 表示向左（用于终点），"right" 表示向右（用于起点）。
        scan_range_px: 向指定方向扫描的像素范围。
        row_span: 扫描线上下扩展行数。
        edge_threshold: 色差梯度阈值。
        cluster_gap: 边缘聚类最大间隔。
        full_ratio_range: 判定为完整色块的宽度比例区间。
        half_ratio_range: 判定为半个色块的宽度比例区间。

    Returns:
        包含以下字段的字典：
        - x, y: 输入坐标。
        - block_width: 当前色块宽度（像素）。
        - std_width: 邻域标准色块宽度（中位数）。
        - ratio: 当前宽度与标准宽度的比例。
        - category: "full"（完整）、"half"（半块）或 "unknown"（未知）。
        - boundaries: 当前色块左右边界 (left, right)。
        - reference_widths: 参考色块宽度列表。
        - message: 诊断信息。

    Raises:
        ValueError: 参数不合法或无法定位到有效色块。
    """
    image = _validate_image(image)
    h, w = image.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        raise ValueError(f"坐标越界: ({x}, {y})，图像尺寸: ({w}, {h})")
    if scan_range_px <= 0:
        raise ValueError("scan_range_px 必须大于 0")
    if direction not in ("left", "right"):
        raise ValueError("direction 必须是 'left' 或 'right'")

    boundaries = extract_horizontal_boundaries(
        image=image,
        y=y,
        row_span=row_span,
        edge_threshold=edge_threshold,
        cluster_gap=cluster_gap,
    )

    segment = _find_containing_segment(boundaries, x, w)
    if segment is None:
        raise ValueError(f"无法为坐标 ({x}, {y}) 定位到有效色块区间")

    idx, left, right = segment
    block_width = right - left

    # 收集参考色块宽度
    reference_widths: List[float] = []
    if direction == "right":
        end_x = min(w - 1, x + scan_range_px)
        # 只取完全落在扫描范围内的色块
        for j in range(idx + 1, boundaries.size - 1):
            b_left = boundaries[j]
            b_right = boundaries[j + 1]
            if b_left >= end_x:
                break
            if b_right <= end_x:
                reference_widths.append(float(b_right - b_left))
    else:  # left
        start_x = max(0, x - scan_range_px)
        for j in range(idx - 1, -1, -1):
            b_left = boundaries[j]
            b_right = boundaries[j + 1]
            if b_right <= start_x:
                break
            if b_left >= start_x:
                reference_widths.append(float(b_right - b_left))

    if not reference_widths:
        return {
            "x": x,
            "y": y,
            "block_width": float(block_width),
            "std_width": 0.0,
            "ratio": 0.0,
            "category": "unknown",
            "boundaries": (left, right),
            "reference_widths": reference_widths,
            "message": "扫描范围内未找到参考色块",
        }

    std_width = float(np.median(reference_widths))
    if std_width <= 0:
        raise ValueError("标准宽度计算异常，请检查边界检测结果")

    ratio = float(block_width / std_width)
    full_lo, full_hi = full_ratio_range
    half_lo, half_hi = half_ratio_range

    if full_lo <= ratio <= full_hi:
        category = "full"
    elif half_lo <= ratio <= half_hi:
        category = "half"
    else:
        category = "unknown"

    return {
        "x": x,
        "y": y,
        "block_width": float(block_width),
        "std_width": std_width,
        "ratio": ratio,
        "category": category,
        "boundaries": (left, right),
        "reference_widths": reference_widths,
        "message": "检测完成",
    }


def analyze_color_bar_endpoints(
    image: np.ndarray,
    start_pos: Tuple[int, int],
    end_pos: Tuple[int, int],
    scan_range_px: int = DEFAULT_SCAN_RANGE_PX,
    row_span: int = DEFAULT_ROW_SPAN,
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
    cluster_gap: int = DEFAULT_CLUSTER_GAP,
    full_ratio_range: Tuple[float, float] = DEFAULT_FULL_RATIO_RANGE,
    half_ratio_range: Tuple[float, float] = DEFAULT_HALF_RATIO_RANGE,
) -> dict:
    """同时判断色标条起点和终点色块宽度。

    Args:
        image: 输入图像。
        start_pos: 起点坐标 (x, y)，将与其右侧色块比较。
        end_pos: 终点坐标 (x, y)，将与其左侧色块比较。
        scan_range_px: 向两侧扫描的像素范围。
        row_span, edge_threshold, cluster_gap: 边界检测参数。
        full_ratio_range, half_ratio_range: 分类阈值。

    Returns:
        包含 start 和 end 两个字典结果的字典。
    """
    start_x, start_y = start_pos
    end_x, end_y = end_pos

    start_result = classify_endpoint_block(
        image=image,
        x=start_x,
        y=start_y,
        direction="right",
        scan_range_px=scan_range_px,
        row_span=row_span,
        edge_threshold=edge_threshold,
        cluster_gap=cluster_gap,
        full_ratio_range=full_ratio_range,
        half_ratio_range=half_ratio_range,
    )

    end_result = classify_endpoint_block(
        image=image,
        x=end_x,
        y=end_y,
        direction="left",
        scan_range_px=scan_range_px,
        row_span=row_span,
        edge_threshold=edge_threshold,
        cluster_gap=cluster_gap,
        full_ratio_range=full_ratio_range,
        half_ratio_range=half_ratio_range,
    )

    return {
        "start": start_result,
        "end": end_result,
    }
