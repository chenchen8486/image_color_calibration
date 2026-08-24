"""block_detector 模块单元测试。"""

import os
import sys
import unittest
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.block_detector import (
    extract_horizontal_boundaries,
    compute_block_widths,
    classify_endpoint_block,
    analyze_color_bar_endpoints,
)


class TestBlockDetector(unittest.TestCase):
    """测试色块宽度检测核心函数。"""

    @staticmethod
    def _make_synthetic_image(
        width: int = 1000,
        block_widths: list[int] | None = None,
    ) -> np.ndarray:
        """构造合成色标条图像。

        Args:
            width: 图像宽度。
            block_widths: 每个色块宽度列表，None 时使用等宽色块。

        Returns:
            RGB 图像数组。
        """
        if block_widths is None:
            block_widths = [80] * 12
        colors = [
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [0, 255, 255],
            [255, 0, 255],
        ]
        img = np.zeros((100, width, 3), dtype=np.uint8)
        x = 0
        for i, bw in enumerate(block_widths):
            color = colors[i % len(colors)]
            img[:, x : x + bw, :] = color
            x += bw
        return img

    def test_extract_horizontal_boundaries_equal_blocks(self) -> None:
        """等宽色块应检测出均匀间隔边界。"""
        img = self._make_synthetic_image(width=960, block_widths=[80] * 12)
        boundaries = extract_horizontal_boundaries(img, y=50)
        widths = compute_block_widths(boundaries)
        self.assertGreaterEqual(len(widths), 10)
        self.assertTrue(np.allclose(widths[:10], 80, atol=2))

    def test_classify_full_start_block(self) -> None:
        """起点在完整色块内应被分类为 full。"""
        img = self._make_synthetic_image(width=960, block_widths=[80] * 12)
        result = classify_endpoint_block(
            image=img,
            x=120,
            y=50,
            direction="right",
            scan_range_px=500,
        )
        self.assertEqual(result["category"], "full")
        self.assertGreater(result["std_width"], 0)
        self.assertGreater(len(result["reference_widths"]), 0)

    def test_classify_half_end_block(self) -> None:
        """终点在半块内应被分类为 half。"""
        block_widths = [80] * 10 + [40]
        width = sum(block_widths)
        img = self._make_synthetic_image(width=width, block_widths=block_widths)
        # 最后一个半块中心约在 820
        result = classify_endpoint_block(
            image=img,
            x=820,
            y=50,
            direction="left",
            scan_range_px=500,
        )
        self.assertEqual(result["category"], "half")
        self.assertAlmostEqual(result["ratio"], 0.5, delta=0.15)

    def test_analyze_color_bar_endpoints(self) -> None:
        """同时分析起点和终点。"""
        block_widths = [80] * 3 + [80] * 8 + [40]
        width = sum(block_widths)
        img = self._make_synthetic_image(width=width, block_widths=block_widths)
        result = analyze_color_bar_endpoints(
            image=img,
            start_pos=(120, 50),
            end_pos=(width - 20, 50),
            scan_range_px=500,
        )
        self.assertEqual(result["start"]["category"], "full")
        self.assertEqual(result["end"]["category"], "half")

    def test_invalid_coordinate_raises(self) -> None:
        """越界坐标应抛出异常。"""
        img = self._make_synthetic_image()
        with self.assertRaises(ValueError):
            classify_endpoint_block(
                image=img,
                x=99999,
                y=50,
                direction="right",
            )


if __name__ == "__main__":
    unittest.main()
