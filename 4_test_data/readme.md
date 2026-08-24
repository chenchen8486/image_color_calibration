# 色标条端点坐标校正工具（C++）

## 1. 项目简介

本项目是 `test.cpp` 对应的说明文档，用于读取色标条扫描图像，检测起点/终点所在色块的宽度，并自动将输入坐标校正到**完整色块的中心**（半块直接抛弃，内移到相邻完整色块）。

**关键语义**：
- **起点坐标**对应色标条**右侧**边界。
- **终点坐标**对应色标条**左侧**边界。

输入为图像路径、起点/终点坐标；输出为调整后的起点/终点坐标。当开启 `debug` 模式时，还会生成与算法流程对应的调试图片，便于定位问题。

---

## 2. 环境依赖

- **OpenCV C++**：需安装 `opencv_core`、`opencv_imgproc`、`opencv_imgcodecs` 模块。
- **C++17**：代码使用了 `std::filesystem` 创建调试输出目录。
- 支持编译器：g++（MinGW/MSYS2）、MSVC（Visual Studio 2017+）、Clang。

---

## 3. 目录结构

```text
4_test_data/
├── test.cpp              # 主程序与核心函数
├── test.bmp              # 输入图像（示例）
├── debug/                # debug=true 时生成的调试图片目录
│   ├── 1_start_brightness_boundaries.bmp
│   ├── 2_start_gradient_edges.bmp
│   ├── 3_start_width_comparison.bmp
│   ├── 1_end_brightness_boundaries.bmp
│   ├── 2_end_gradient_edges.bmp
│   └── 3_end_width_comparison.bmp
└── 算法流程文档.md        # 本文档
```

---

## 4. 核心接口

```cpp
// 坐标结构体
struct PointP {
    int x;
    int y;
};

/**
 * 重新计算并调整色标条起点和终点坐标。
 *
 * 注意：色标条右侧为起点坐标，左侧为终点坐标。
 *
 * @param imgPath                 图像路径。
 * @param point_start             起点坐标（色标条右侧）。
 * @param point_end               终点坐标（色标条左侧）。
 * @param processed_point_start   输出调整后的起点坐标。
 * @param processed_point_end     输出调整后的终点坐标。
 * @param debug                   是否输出 debug 调试图片，默认 false。
 * @param debugOutputDir          debug 图片输出目录，默认 "./debug"。
 */
void recaluate_coordinates(
    std::string imgPath,
    PointP point_start,
    PointP point_end,
    PointP& processed_point_start,
    PointP& processed_point_end,
    bool debug = false,
    std::string debugOutputDir = "./debug"
);
```

**坐标调整规则**：
- 一律返回**完整色块的中心 x 坐标**（y 坐标保持输入不变）。
- 起点在色标条**右侧**：所在色块完整 → 返回其中心；为**半块** → 抛弃之，**左移**到相邻完整色块并返回其中心。
- 终点在色标条**左侧**：所在色块完整 → 返回其中心；为**半块** → 抛弃之，**右移**到相邻完整色块并返回其中心。
- 业务约定：半块只可能出现在最外侧第一个色块，其余色块必定完整，因此相邻色块无需再校验。
- 分类为 `unknown`（无参考色块或比例不在判定区间内）时**抛出异常**，对应输出坐标回退为原始输入坐标。

---

## 5. 算法流程

| 步骤 | 说明 | 对应 C++ 函数/代码 |
|---|---|---|
| 1 | **图像读取**：使用 OpenCV 读取图像，BGR → RGB，转换为 `CV_32F`。 | `cv::imread`、`cv::cvtColor`、`convertTo` |
| 2 | **扫描线提取与平滑**：以 `y` 为中心，向上下各扩展 `row_span` 行，对多行像素在 Y 方向求平均，得到平滑颜色曲线。 | `smoothScanLine` |
| 3 | **色差梯度计算**：对平滑曲线逐像素计算相邻点 L2 欧氏距离，得到梯度 `G[x]`。 | `computeGradient` |
| 4 | **候选边缘提取**：梯度超过 `edge_threshold` 的位置标记为候选边缘。 | `extractHorizontalBoundaries` 前半段 |
| 5 | **边缘聚类**：将间隔不超过 `cluster_gap` 的候选边缘归为一组，取均值作为真实边界。 | `clusterEdges` |
| 6 | **定位色块区间**：找到输入坐标 `x` 所在的色块区间 `(left, right)`，支持图像边缘情况。 | `findContainingSegment` |
| 7 | **计算当前色块宽度**：`block_width = right - left`。 | `classifyAndAdjust` |
| 8 | **收集参考宽度**：向指定方向扫描 `scan_range_px` 像素，收集完全落在范围内的邻域色块宽度。 | `classifyAndAdjust` |
| 9 | **标准宽度与比例**：取参考宽度的中位数作为标准宽度，计算 `ratio = block_width / std_width`。 | `median`、`classifyAndAdjust` |
| 10 | **分类判定**：根据比例区间判定为 `full`（完整）、`half`（半块）或 `unknown`（未知）。 | `classifyAndAdjust` |
| 11 | **坐标调整**：完整块返回其中心；半块抛弃并内移到相邻完整色块的中心；unknown 抛异常并回退为原始坐标。 | `classifyAndAdjust`、`recaluate_coordinates` |

---

## 6. 参数说明

| 参数/常量 | 默认值 | 含义 |
|---|---|---|
| `DEFAULT_SCAN_RANGE_PX` | `1000` | 向单侧扫描的像素范围，用于收集参考色块。 |
| `DEFAULT_ROW_SPAN` | `10` | 扫描线上下扩展行数，对 `[y-row_span, y+row_span]` 行做平均降噪。 |
| `DEFAULT_EDGE_THRESHOLD` | `20.0` | 色差梯度阈值，相邻像素 L2 色差超过此值视为候选边缘。 |
| `DEFAULT_CLUSTER_GAP` | `3` | 边缘聚类最大像素间隔，间隔不超过此值的候选边缘合并为一个真实边界。 |
| `FULL_RATIO_LO / FULL_RATIO_HI` | `0.75 / 1.25` | 判定为完整色块的宽度比例区间。 |
| `HALF_RATIO_LO / HALF_RATIO_HI` | `0.35 / 0.65` | 判定为半块的宽度比例区间。 |

---

## 7. Debug 输出图片

开启 `debug = true` 后，会在 `debugOutputDir` 目录下生成以下图片，每张图片对应算法流程中的关键步骤。

| 文件名 | 对应步骤 | 图中颜色/元素含义 |
|---|---|---|
| `1_start_brightness_boundaries.bmp` | 步骤 1~2、5~6 | 黑色曲线：平滑后亮度；红色竖线：检测点；蓝色竖线：聚类边界；绿色半透明区域：当前色块区间。 |
| `2_start_gradient_edges.bmp` | 步骤 3~5 | 深橙色曲线：色差梯度 `G[x]`；紫色横线：阈值 `T`；红色圆点：候选边缘；蓝色竖线：真实边界。 |
| `3_start_width_comparison.bmp` | 步骤 7~10 | 绿色柱子：当前色块宽度；蓝色柱子：参考色块宽度；红色横线：标准宽度（中位数）。 |
| `1_end_brightness_boundaries.bmp` | 步骤 1~2、5~6 | 同上，针对终点。 |
| `2_end_gradient_edges.bmp` | 步骤 3~5 | 同上，针对终点。 |
| `3_end_width_comparison.bmp` | 步骤 7~10 | 同上，针对终点。 |

> 说明：调试图片使用 OpenCV 原生绘图，标签为英文/ASCII，避免中文字体依赖。

---

## 8. 编译与运行

### 使用 g++（MinGW / MSYS2）

```bash
g++ -std=c++17 test.cpp -o test.exe \
    -I<opencv_include路径> \
    -L<opencv_lib路径> \
    -lopencv_core -lopencv_imgproc -lopencv_imgcodecs
```

### 使用 Visual Studio

1. 新建空项目，将 `test.cpp` 添加到源文件。
2. 项目属性 → VC++ 目录：配置 OpenCV 的「包含目录」和「库目录」。
3. 项目属性 → 链接器 → 输入：添加以下依赖项：
   - `opencv_corexxx.lib`
   - `opencv_imgprocxxx.lib`
   - `opencv_imgcodecsxxx.lib`
4. 确保 C++ 语言标准设置为 **C++17**。

### 运行

```bash
./test.exe
```

### 期望输出

```text
起点分类: full, 比例: 0.98
终点分类: half, 比例: 0.52
调整后起点: (13985, 490)
调整后终点: (40, 490)
```

同时会在 `./debug` 目录下生成 6 张 `.bmp` 调试图片。

---

## 9. 使用示例

```cpp
#include <string>
#include <iostream>

// 假设 test.cpp 中的 PointP 和 recaluate_coordinates 已可见

int main() {
    std::string imgPath = "test.bmp";

    // 注意：右侧为起点，左侧为终点
    PointP start = {13983, 490}; // 色标条右侧起点
    PointP end   = {42, 490};    // 色标条左侧终点

    PointP processedStart, processedEnd;

    // 开启 debug，输出调试图片
    recaluate_coordinates(
        imgPath, start, end,
        processedStart, processedEnd,
        true, "./debug"
    );

    std::cout << "调整后起点: ("
              << processedStart.x << ", " << processedStart.y << ")" << std::endl;
    std::cout << "调整后终点: ("
              << processedEnd.x << ", " << processedEnd.y << ")" << std::endl;

    return 0;
}
```

---

## 10. 注意事项

1. **起点/终点语义**：
   - 起点 = 色标条**右侧**边界。
   - 终点 = 色标条**左侧**边界。
   - 与 Python 脚本中的命名相反，调用时务必注意。

2. **坐标越界**：输入坐标必须在图像宽高范围内，否则会抛出异常。

3. **OpenCV 颜色空间**：内部统一按 RGB 处理，读取时已完成 BGR → RGB 转换。

4. **边界取整**：输出坐标为整数，对浮点边界进行四舍五入。

5. **分类为 unknown**：如果扫描范围内没有收集到参考色块，或宽度比例不在完整/半块判定区间内，分类为 `unknown`，`classifyAndAdjust` 抛出异常；`recaluate_coordinates` 捕获后将对应输出坐标**回退为原始输入坐标**并继续向上抛出。

6. **调试目录**：使用 `std::filesystem::create_directories` 自动创建目录，需 C++17 支持。

---

## 11. 变更记录

| 时间 | 内容 |
|---|---|
| 2026-07-24 | 初始版本，完成 C++ 代码与 README 文档。 |
| 2026-07-24 | 添加 `debug` 模式与对应调试图片输出说明。 |
| 2026-07-24 | 坐标校正规则变更：由 snap 到色块边界改为返回完整色块中心；半块抛弃并内移到相邻完整色块；unknown 抛异常并回退原始坐标。 |
