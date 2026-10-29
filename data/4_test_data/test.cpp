#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <filesystem>

// 坐标结构体
struct PointP {
    int x;
    int y;
};

// ============================================================================
// 默认参数配置
// ============================================================================

// 向单侧扫描的像素范围，用于收集参考色块
const int DEFAULT_SCAN_RANGE_PX = 1000;

// 扫描线上下扩展行数，对 [y-row_span, y+row_span] 行做平均以降低单行噪声
const int DEFAULT_ROW_SPAN = 10;

// 色差梯度阈值，相邻像素 L2 色差超过此值视为候选边缘
const float DEFAULT_EDGE_THRESHOLD = 20.0f;

// 边缘聚类最大像素间隔，间隔不超过此值的候选边缘合并为一个真实边界
const int DEFAULT_CLUSTER_GAP = 3;

// 判定为完整色块的宽度比例区间 [lo, hi]
const float FULL_RATIO_LO = 0.75f;
const float FULL_RATIO_HI = 1.25f;

// 判定为半块的宽度比例区间 [lo, hi]
const float HALF_RATIO_LO = 0.35f;
const float HALF_RATIO_HI = 0.65f;

// ============================================================================
// 绘图配置（debug 输出用）
// ============================================================================

struct PlotConfig {
    int width = 1200;          // 画布宽度（像素）
    int height = 600;          // 画布高度（像素）
    int marginLeft = 80;       // 左边距
    int marginRight = 40;      // 右边距
    int marginTop = 60;        // 上边距
    int marginBottom = 80;     // 下边距
    cv::Scalar bgColor = cv::Scalar(255, 255, 255);   // 白色背景
    cv::Scalar axisColor = cv::Scalar(0, 0, 0);       // 黑色坐标轴
};

// ============================================================================
// 基础算法函数
// ============================================================================

/**
 * 提取并平滑扫描线颜色曲线。
 *
 * @param image 输入图像，CV_32F 三通道。
 * @param y 扫描线中心行。
 * @param rowSpan 扫描线上下扩展行数。
 * @return 形状为 (1, W, 3) 的 float32 曲线。
 */
cv::Mat smoothScanLine(const cv::Mat& image, int y, int rowSpan) {
    int h = image.rows;
    int y0 = std::max(0, y - rowSpan);
    int y1 = std::min(h, y + rowSpan + 1);
    cv::Mat strip = image(cv::Range(y0, y1), cv::Range::all());
    cv::Mat curve;
    cv::reduce(strip, curve, 0, cv::REDUCE_AVG, CV_32F);
    return curve;
}

/**
 * 计算相邻像素 L2 色差梯度。
 *
 * @param curve 平滑后的颜色曲线。
 * @return 梯度数组，长度为 W-1。
 */
std::vector<float> computeGradient(const cv::Mat& curve) {
    int w = curve.cols;
    std::vector<float> grad;
    grad.reserve(w - 1);
    for (int x = 0; x < w - 1; ++x) {
        cv::Vec3f c1 = curve.at<cv::Vec3f>(0, x);
        cv::Vec3f c2 = curve.at<cv::Vec3f>(0, x + 1);
        float dx = c1[0] - c2[0];
        float dy = c1[1] - c2[1];
        float dz = c1[2] - c2[2];
        grad.push_back(std::sqrt(dx * dx + dy * dy + dz * dz));
    }
    return grad;
}

/**
 * 将邻近边缘聚类为真实边界。
 *
 * @param edges 候选边缘 x 坐标数组。
 * @param clusterGap 同组边缘最大像素间隔。
 * @return 聚类后的边界位置数组。
 */
std::vector<float> clusterEdges(const std::vector<int>& edges, int clusterGap) {
    if (edges.empty()) {
        return {};
    }
    std::vector<float> boundaries;
    int groupStart = edges[0];
    int groupEnd = groupStart;
    for (size_t i = 1; i < edges.size(); ++i) {
        int x = edges[i];
        if (x - groupEnd <= clusterGap) {
            groupEnd = x;
        } else {
            boundaries.push_back((groupStart + groupEnd) / 2.0f);
            groupStart = x;
            groupEnd = x;
        }
    }
    boundaries.push_back((groupStart + groupEnd) / 2.0f);
    return boundaries;
}

/**
 * 提取水平方向色块边界。
 *
 * @param image 输入图像。
 * @param y 扫描线中心 y 坐标。
 * @param rowSpan 扫描线上下扩展行数。
 * @param edgeThreshold 相邻像素色差阈值。
 * @param clusterGap 边缘聚类最大像素间隔。
 * @return 边界位置数组。
 */
std::vector<float> extractHorizontalBoundaries(
    const cv::Mat& image,
    int y,
    int rowSpan,
    float edgeThreshold,
    int clusterGap
) {
    cv::Mat curve = smoothScanLine(image, y, rowSpan);
    std::vector<float> grad = computeGradient(curve);
    std::vector<int> edges;
    for (size_t i = 0; i < grad.size(); ++i) {
        if (grad[i] > edgeThreshold) {
            edges.push_back(static_cast<int>(i));
        }
    }
    return clusterEdges(edges, clusterGap);
}

/**
 * 找到 x 所在的色块区间。
 *
 * @param boundaries 已排序的边界位置数组。
 * @param x 待查询 x 坐标。
 * @param imageWidth 图像宽度。
 * @param idx 输出区间索引。
 * @param left 输出左边界。
 * @param right 输出右边界。
 * @return 是否找到有效区间。
 */
bool findContainingSegment(
    const std::vector<float>& boundaries,
    int x,
    int imageWidth,
    int& idx,
    float& left,
    float& right
) {
    if (boundaries.empty()) {
        return false;
    }
    float first = boundaries.front();
    float last = boundaries.back();

    // 位于第一个边界左侧，将 0 作为左边界
    if (x <= first) {
        if (0 <= x && x <= first) {
            idx = -1;
            left = 0.0f;
            right = first;
            return true;
        }
        return false;
    }

    // 位于最后一个边界右侧，将图像右边缘作为右边界
    if (x >= last) {
        if (last <= x && x < imageWidth) {
            idx = static_cast<int>(boundaries.size()) - 1;
            left = last;
            right = static_cast<float>(imageWidth - 1);
            return true;
        }
        return false;
    }

    // 位于中间某两个边界之间
    auto it = std::upper_bound(boundaries.begin(), boundaries.end(), static_cast<float>(x));
    if (it == boundaries.begin() || it == boundaries.end()) {
        return false;
    }
    idx = static_cast<int>(it - boundaries.begin()) - 1;
    left = boundaries[idx];
    right = boundaries[idx + 1];
    if (left <= x && x <= right) {
        return true;
    }
    return false;
}

/**
 * 计算中位数。
 */
float median(std::vector<float> values) {
    if (values.empty()) {
        return 0.0f;
    }
    size_t n = values.size();
    std::sort(values.begin(), values.end());
    if (n % 2 == 1) {
        return values[n / 2];
    }
    return (values[n / 2 - 1] + values[n / 2]) / 2.0f;
}

// ============================================================================
// Debug 绘图辅助函数
// ============================================================================

/**
 * 将数据坐标映射到画布像素坐标。
 */
cv::Point mapToCanvas(
    float x, float y,
    float xMin, float xMax,
    float yMin, float yMax,
    const PlotConfig& cfg
) {
    int plotW = cfg.width - cfg.marginLeft - cfg.marginRight;
    int plotH = cfg.height - cfg.marginTop - cfg.marginBottom;
    int px = cfg.marginLeft + static_cast<int>((x - xMin) / (xMax - xMin) * plotW);
    int py = cfg.height - cfg.marginBottom - static_cast<int>((y - yMin) / (yMax - yMin) * plotH);
    return cv::Point(px, py);
}

/**
 * 绘制折线。
 */
void drawPolyline(
    cv::Mat& canvas,
    const std::vector<float>& xs,
    const std::vector<float>& ys,
    float xMin, float xMax,
    float yMin, float yMax,
    const cv::Scalar& color,
    int thickness,
    const PlotConfig& cfg
) {
    if (xs.size() < 2 || ys.size() < 2) return;
    for (size_t i = 0; i < xs.size() - 1; ++i) {
        cv::Point p1 = mapToCanvas(xs[i], ys[i], xMin, xMax, yMin, yMax, cfg);
        cv::Point p2 = mapToCanvas(xs[i + 1], ys[i + 1], xMin, xMax, yMin, yMax, cfg);
        cv::line(canvas, p1, p2, color, thickness);
    }
}

/**
 * 绘制垂直参考线。
 */
void drawVerticalLine(
    cv::Mat& canvas,
    float x,
    float xMin, float xMax,
    float yMin, float yMax,
    const cv::Scalar& color,
    int thickness,
    const PlotConfig& cfg
) {
    cv::Point p1 = mapToCanvas(x, yMin, xMin, xMax, yMin, yMax, cfg);
    cv::Point p2 = mapToCanvas(x, yMax, xMin, xMax, yMin, yMax, cfg);
    cv::line(canvas, p1, p2, color, thickness);
}

/**
 * 绘制半透明矩形区域（用于高亮当前色块）。
 */
void drawShadedRegion(
    cv::Mat& canvas,
    float xLeft, float xRight,
    float xMin, float xMax,
    float yMin, float yMax,
    const cv::Scalar& color,
    float alpha,
    const PlotConfig& cfg
) {
    cv::Point tl = mapToCanvas(xLeft, yMax, xMin, xMax, yMin, yMax, cfg);
    cv::Point br = mapToCanvas(xRight, yMin, xMin, xMax, yMin, yMax, cfg);
    // 色块区间可能超出绘图窗口（如检测点位于首/尾色块），需裁剪到画布内，
    // 否则 canvas(roi) 越界会触发 OpenCV 断言导致程序崩溃。
    cv::Rect roi = cv::Rect(tl, br) & cv::Rect(0, 0, canvas.cols, canvas.rows);
    if (roi.width <= 0 || roi.height <= 0) return;

    cv::Mat overlay = canvas(roi).clone();
    cv::rectangle(overlay, cv::Point(0, 0), cv::Point(roi.width, roi.height), color, -1);
    cv::addWeighted(overlay, alpha, canvas(roi), 1.0 - alpha, 0.0, canvas(roi));
}

/**
 * 将曲线数据转换为 x/y 数组。
 */
void curveToXY(const cv::Mat& curve, int x0, std::vector<float>& xs, std::vector<float>& ys) {
    xs.clear();
    ys.clear();
    for (int x = 0; x < curve.cols; ++x) {
        cv::Vec3f c = curve.at<cv::Vec3f>(0, x);
        xs.push_back(static_cast<float>(x0 + x));
        ys.push_back((c[0] + c[1] + c[2]) / 3.0f);
    }
}

/**
 * 绘制 Debug 图 1：扫描线亮度曲线 + 聚类边界 + 当前色块高亮。
 *
 * 对应算法步骤：步骤 1（扫描线提取）、步骤 2（平滑降噪）、步骤 5（边缘聚类）、步骤 6（色块定位）。
 */
void drawBrightnessAndBoundaries(
    const cv::Mat& image,
    int y,
    int rowSpan,
    int x,
    int x0,
    int x1,
    const std::vector<float>& boundaries,
    float left,
    float right,
    const std::string& prefix,
    const std::string& outputPath
) {
    cv::Mat curve = smoothScanLine(image, y, rowSpan);
    std::vector<float> xs, ys;
    curveToXY(curve, 0, xs, ys);

    // 只取窗口内的数据用于绘图
    std::vector<float> winXs, winYs;
    for (size_t i = 0; i < xs.size(); ++i) {
        if (xs[i] >= x0 && xs[i] <= x1) {
            winXs.push_back(xs[i]);
            winYs.push_back(ys[i]);
        }
    }
    if (winYs.empty()) return;

    float yMin = *std::min_element(winYs.begin(), winYs.end()) * 0.95f;
    float yMax = *std::max_element(winYs.begin(), winYs.end()) * 1.05f;
    if (yMax - yMin < 1e-6f) {
        yMax = yMin + 1.0f;
    }

    PlotConfig cfg;
    cv::Mat canvas(cfg.height, cfg.width, CV_8UC3, cfg.bgColor);

    // 绘制坐标轴
    cv::line(canvas,
             cv::Point(cfg.marginLeft, cfg.height - cfg.marginBottom),
             cv::Point(cfg.width - cfg.marginRight, cfg.height - cfg.marginBottom),
             cfg.axisColor, 1);
    cv::line(canvas,
             cv::Point(cfg.marginLeft, cfg.marginTop),
             cv::Point(cfg.marginLeft, cfg.height - cfg.marginBottom),
             cfg.axisColor, 1);

    // 标题与标签
    std::string title = prefix + " Step 1-2: Smoothed Brightness (y=" + std::to_string(y) + ")";
    cv::putText(canvas, title,
                cv::Point(cfg.marginLeft, cfg.marginTop - 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cfg.axisColor, 2);
    cv::putText(canvas, "x (pixel)",
                cv::Point(cfg.width - cfg.marginRight - 80, cfg.height - 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cfg.axisColor, 1);
    cv::putText(canvas, "brightness",
                cv::Point(15, cfg.marginTop + 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cfg.axisColor, 1);

    // 高亮当前色块区域（绿色半透明）
    drawShadedRegion(canvas, left, right, static_cast<float>(x0), static_cast<float>(x1),
                     yMin, yMax, cv::Scalar(0, 255, 0), 0.15f, cfg);

    // 绘制亮度曲线（黑色）
    drawPolyline(canvas, winXs, winYs, static_cast<float>(x0), static_cast<float>(x1),
                 yMin, yMax, cv::Scalar(0, 0, 0), 1, cfg);

    // 绘制检测点（红色虚线效果：用密集实线近似）
    drawVerticalLine(canvas, static_cast<float>(x), static_cast<float>(x0), static_cast<float>(x1),
                     yMin, yMax, cv::Scalar(0, 0, 255), 2, cfg);

    // 绘制窗口内的聚类边界（蓝色）
    for (float b : boundaries) {
        if (b >= x0 && b <= x1) {
            drawVerticalLine(canvas, b, static_cast<float>(x0), static_cast<float>(x1),
                             yMin, yMax, cv::Scalar(255, 0, 0), 1, cfg);
        }
    }

    // 图例
    cv::putText(canvas, "red=detection point",
                cv::Point(cfg.marginLeft, cfg.marginTop + 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
    cv::putText(canvas, "blue=clustered boundary",
                cv::Point(cfg.marginLeft, cfg.marginTop + 50),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
    cv::putText(canvas, "green=current block",
                cv::Point(cfg.marginLeft, cfg.marginTop + 70),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 200, 0), 1);

    cv::imwrite(outputPath, canvas);
}

/**
 * 绘制 Debug 图 2：色差梯度 + 阈值 + 候选边缘 + 聚类边界。
 *
 * 对应算法步骤：步骤 3（梯度计算）、步骤 4（阈值提取）、步骤 5（边缘聚类）。
 */
void drawGradientAndEdges(
    const cv::Mat& image,
    int y,
    int rowSpan,
    float edgeThreshold,
    int clusterGap,
    int x,
    int x0,
    int x1,
    const std::vector<float>& boundaries,
    const std::string& prefix,
    const std::string& outputPath
) {
    cv::Mat curve = smoothScanLine(image, y, rowSpan);
    std::vector<float> grad = computeGradient(curve);

    std::vector<float> xs, ys;
    for (size_t i = 0; i < grad.size(); ++i) {
        xs.push_back(static_cast<float>(i));
        ys.push_back(grad[i]);
    }

    std::vector<float> winXs, winYs;
    for (size_t i = 0; i < xs.size(); ++i) {
        if (xs[i] >= x0 && xs[i] <= x1) {
            winXs.push_back(xs[i]);
            winYs.push_back(ys[i]);
        }
    }
    if (winYs.empty()) return;

    float yMin = 0.0f;
    float yMax = *std::max_element(winYs.begin(), winYs.end()) * 1.1f;
    if (yMax < edgeThreshold * 1.2f) {
        yMax = edgeThreshold * 1.2f;
    }

    PlotConfig cfg;
    cv::Mat canvas(cfg.height, cfg.width, CV_8UC3, cfg.bgColor);

    // 坐标轴
    cv::line(canvas,
             cv::Point(cfg.marginLeft, cfg.height - cfg.marginBottom),
             cv::Point(cfg.width - cfg.marginRight, cfg.height - cfg.marginBottom),
             cfg.axisColor, 1);
    cv::line(canvas,
             cv::Point(cfg.marginLeft, cfg.marginTop),
             cv::Point(cfg.marginLeft, cfg.height - cfg.marginBottom),
             cfg.axisColor, 1);

    // 标题与标签
    std::string title = prefix + " Step 3-4: Gradient, Threshold and Edges";
    cv::putText(canvas, title,
                cv::Point(cfg.marginLeft, cfg.marginTop - 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cfg.axisColor, 2);
    cv::putText(canvas, "x (pixel)",
                cv::Point(cfg.width - cfg.marginRight - 80, cfg.height - 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cfg.axisColor, 1);
    cv::putText(canvas, "gradient",
                cv::Point(15, cfg.marginTop + 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cfg.axisColor, 1);

    // 绘制梯度曲线（深橙色）
    drawPolyline(canvas, winXs, winYs, static_cast<float>(x0), static_cast<float>(x1),
                 yMin, yMax, cv::Scalar(0, 140, 255), 2, cfg);

    // 绘制阈值线（紫色）
    std::vector<float> thXs = {static_cast<float>(x0), static_cast<float>(x1)};
    std::vector<float> thYs = {edgeThreshold, edgeThreshold};
    drawPolyline(canvas, thXs, thYs, static_cast<float>(x0), static_cast<float>(x1),
                 yMin, yMax, cv::Scalar(255, 0, 255), 2, cfg);

    // 候选边缘点（红色）
    for (size_t i = 0; i < grad.size(); ++i) {
        if (grad[i] > edgeThreshold && static_cast<int>(i) >= x0 && static_cast<int>(i) <= x1) {
            cv::Point p = mapToCanvas(static_cast<float>(i), grad[i], static_cast<float>(x0), static_cast<float>(x1),
                                      yMin, yMax, cfg);
            cv::circle(canvas, p, 3, cv::Scalar(0, 0, 255), -1);
        }
    }

    // 聚类边界（蓝色）
    for (float b : boundaries) {
        if (b >= x0 && b <= x1) {
            drawVerticalLine(canvas, b, static_cast<float>(x0), static_cast<float>(x1),
                             yMin, yMax, cv::Scalar(255, 0, 0), 1, cfg);
        }
    }

    // 图例
    cv::putText(canvas, "orange=gradient G[x]",
                cv::Point(cfg.marginLeft, cfg.marginTop + 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 140, 255), 1);
    cv::putText(canvas, "purple=threshold T",
                cv::Point(cfg.marginLeft, cfg.marginTop + 50),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 255), 1);
    cv::putText(canvas, "red=candidate edge",
                cv::Point(cfg.marginLeft, cfg.marginTop + 70),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
    cv::putText(canvas, "blue=clustered boundary",
                cv::Point(cfg.marginLeft, cfg.marginTop + 90),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);

    cv::imwrite(outputPath, canvas);
}

/**
 * 绘制 Debug 图 3：当前色块与参考色块宽度对比柱状图。
 *
 * 对应算法步骤：步骤 7（计算宽度）、步骤 8（收集参考）、步骤 9（标准宽度）、步骤 10（分类判定）。
 */
void drawWidthComparison(
    float currentWidth,
    const std::vector<float>& refWidths,
    float stdWidth,
    const std::string& prefix,
    const std::string& outputPath
) {
    PlotConfig cfg;
    cfg.height = 700;
    cv::Mat canvas(cfg.height, cfg.width, CV_8UC3, cfg.bgColor);

    // 标题
    std::string title = prefix + " Step 5-7,10: Block Width Comparison";
    cv::putText(canvas, title,
                cv::Point(cfg.marginLeft, cfg.marginTop - 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cfg.axisColor, 2);

    // 准备数据
    std::vector<float> values;
    std::vector<std::string> labels;
    values.push_back(currentWidth);
    labels.push_back("current");
    for (size_t i = 0; i < refWidths.size(); ++i) {
        values.push_back(refWidths[i]);
        labels.push_back("ref" + std::to_string(i + 1));
    }

    if (values.empty()) return;

    float yMax = *std::max_element(values.begin(), values.end()) * 1.2f;
    if (yMax < 1.0f) yMax = 1.0f;

    // 坐标轴
    cv::line(canvas,
             cv::Point(cfg.marginLeft, cfg.height - cfg.marginBottom),
             cv::Point(cfg.width - cfg.marginRight, cfg.height - cfg.marginBottom),
             cfg.axisColor, 1);
    cv::line(canvas,
             cv::Point(cfg.marginLeft, cfg.marginTop),
             cv::Point(cfg.marginLeft, cfg.height - cfg.marginBottom),
             cfg.axisColor, 1);

    // 绘制柱状图
    int plotW = cfg.width - cfg.marginLeft - cfg.marginRight;
    int barWidth = static_cast<int>(plotW / static_cast<float>(values.size() + 1));
    if (barWidth > 100) barWidth = 100;
    int gap = barWidth / 2;

    for (size_t i = 0; i < values.size(); ++i) {
        int bx = cfg.marginLeft + gap + static_cast<int>(i) * (barWidth + gap);
        int by = cfg.height - cfg.marginBottom - static_cast<int>(values[i] / yMax * (cfg.height - cfg.marginTop - cfg.marginBottom));
        int bh = cfg.height - cfg.marginBottom - by;

        cv::Scalar color = (i == 0) ? cv::Scalar(0, 200, 0) : cv::Scalar(200, 100, 0);
        cv::Rect bar(bx, by, barWidth, bh);
        cv::rectangle(canvas, bar, color, -1);
        cv::rectangle(canvas, bar, cfg.axisColor, 1);

        // 标签
        cv::putText(canvas, labels[i],
                    cv::Point(bx, cfg.height - cfg.marginBottom + 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cfg.axisColor, 1);
        // 数值
        std::string valStr = std::to_string(values[i]);
        valStr = valStr.substr(0, valStr.find('.') + 2);
        cv::putText(canvas, valStr,
                    cv::Point(bx, by - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cfg.axisColor, 1);
    }

    // 标准宽度参考线（红色）
    if (stdWidth > 0.0f) {
        int sy = cfg.height - cfg.marginBottom - static_cast<int>(stdWidth / yMax * (cfg.height - cfg.marginTop - cfg.marginBottom));
        cv::line(canvas,
                 cv::Point(cfg.marginLeft, sy),
                 cv::Point(cfg.width - cfg.marginRight, sy),
                 cv::Scalar(0, 0, 255), 2);
        std::string label = "std=" + std::to_string(stdWidth);
        label = label.substr(0, label.find('.') + 2);
        cv::putText(canvas, label,
                    cv::Point(cfg.width - cfg.marginRight - 100, sy - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
    }

    // 图例
    cv::putText(canvas, "green=current block",
                cv::Point(cfg.marginLeft, cfg.marginTop + 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 200, 0), 1);
    cv::putText(canvas, "blue=reference block",
                cv::Point(cfg.marginLeft, cfg.marginTop + 50),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 100, 0), 1);
    cv::putText(canvas, "red=standard width (median)",
                cv::Point(cfg.marginLeft, cfg.marginTop + 70),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);

    cv::imwrite(outputPath, canvas);
}

/**
 * 根据方向计算扫描窗口 [x0, x1]。
 */
void computeWindow(int x, const std::string& direction, int scanRangePx, int imageWidth, int& x0, int& x1) {
    if (direction == "right") {
        x0 = std::max(0, x - scanRangePx);
        x1 = x;
    } else {
        x0 = x;
        x1 = std::min(imageWidth - 1, x + scanRangePx);
    }
}

// ============================================================================
// 核心检测与调整函数
// ============================================================================

/**
 * 判断指定坐标所在色块宽度，并返回调整后的 x 坐标。
 *
 * @param image 输入图像。
 * @param x 待判断点 x 坐标。
 * @param y 待判断点 y 坐标。
 * @param direction 扫描方向，"left" 表示向左，"right" 表示向右。
 * @param scanRangePx 向指定方向扫描的像素范围。
 * @param rowSpan 扫描线上下扩展行数。
 * @param edgeThreshold 色差梯度阈值。
 * @param clusterGap 边缘聚类最大间隔。
 * @param debug 是否输出 debug 图片。
 * @param debugPrefix debug 图片文件名前缀，如 "start" 或 "end"。
 * @param debugOutputDir debug 图片输出目录。
 * @param outRatio 输出当前宽度与标准宽度的比例。
 * @param outCategory 输出分类字符串。
 * @return 校正后的 x 坐标：当前色块完整时返回其中心；当前色块为半块时抛弃之，
 *         返回相邻完整色块的中心（业务约定：半块只可能出现在最外侧第一个色块，
 *         其相邻色块必定完整）。
 * @throws std::runtime_error 分类为 unknown（无参考色块或比例不在判定区间内）时抛出。
 */
float classifyAndAdjust(
    const cv::Mat& image,
    int x,
    int y,
    const std::string& direction,
    int scanRangePx,
    int rowSpan,
    float edgeThreshold,
    int clusterGap,
    bool debug,
    const std::string& debugPrefix,
    const std::string& debugOutputDir,
    float& outRatio,
    std::string& outCategory
) {
    int w = image.cols;
    int h = image.rows;
    if (x < 0 || x >= w || y < 0 || y >= h) {
        throw std::invalid_argument("坐标越界");
    }

    std::vector<float> boundaries = extractHorizontalBoundaries(
        image, y, rowSpan, edgeThreshold, clusterGap
    );

    int idx;
    float left, right;
    if (!findContainingSegment(boundaries, x, w, idx, left, right)) {
        throw std::runtime_error("无法为坐标定位到有效色块区间");
    }

    float blockWidth = right - left;

    // 收集参考色块宽度
    std::vector<float> refWidths;
    if (direction == "right") {
        int endX = std::min(w - 1, x + scanRangePx);
        for (int j = idx + 1; j < static_cast<int>(boundaries.size()) - 1; ++j) {
            float bl = boundaries[j];
            float br = boundaries[j + 1];
            if (bl >= endX) {
                break;
            }
            if (br <= endX) {
                refWidths.push_back(br - bl);
            }
        }
    } else { // left
        int startX = std::max(0, x - scanRangePx);
        for (int j = idx - 1; j >= 0; --j) {
            float bl = boundaries[j];
            float br = boundaries[j + 1];
            if (br <= startX) {
                break;
            }
            if (bl >= startX) {
                refWidths.push_back(br - bl);
            }
        }
    }

    // Debug 绘图
    if (debug) {
        std::filesystem::create_directories(debugOutputDir);
        int x0, x1;
        computeWindow(x, direction, scanRangePx, w, x0, x1);

        drawBrightnessAndBoundaries(
            image, y, rowSpan, x, x0, x1, boundaries, left, right,
            debugPrefix,
            debugOutputDir + "/1_" + debugPrefix + "_brightness_boundaries.bmp"
        );
        drawGradientAndEdges(
            image, y, rowSpan, edgeThreshold, clusterGap, x, x0, x1, boundaries,
            debugPrefix,
            debugOutputDir + "/2_" + debugPrefix + "_gradient_edges.bmp"
        );
    }

    if (refWidths.empty()) {
        outRatio = 0.0f;
        outCategory = "unknown";
        if (debug) {
            drawWidthComparison(
                blockWidth, refWidths, 0.0f,
                debugPrefix,
                debugOutputDir + "/3_" + debugPrefix + "_width_comparison.bmp"
            );
        }
        throw std::runtime_error("扫描范围内未收集到参考色块，分类未知（unknown），无法校正");
    }

    float stdWidth = median(refWidths);
    if (stdWidth <= 0.0f) {
        throw std::runtime_error("标准宽度计算异常");
    }

    float ratio = blockWidth / stdWidth;
    outRatio = ratio;

    if (FULL_RATIO_LO <= ratio && ratio <= FULL_RATIO_HI) {
        outCategory = "full";
    } else if (HALF_RATIO_LO <= ratio && ratio <= HALF_RATIO_HI) {
        outCategory = "half";
    } else {
        outCategory = "unknown";
    }

    if (debug) {
        drawWidthComparison(
            blockWidth, refWidths, stdWidth,
            debugPrefix,
            debugOutputDir + "/3_" + debugPrefix + "_width_comparison.bmp"
        );
    }

    if (outCategory == "unknown") {
        throw std::runtime_error(
            "色块宽度比例 " + std::to_string(ratio) +
            " 不在完整/半块判定区间内，分类未知（unknown），无法校正"
        );
    }

    // 坐标校正：一律返回完整色块的中心 x 坐标。
    // - full：返回当前色块中心；
    // - half：抛弃当前半块，内移到相邻完整色块并返回其中心。
    //   起点在色标条右侧（direction == "left"），左移；终点在左侧，右移。
    float centerX;
    if (outCategory == "full") {
        centerX = (left + right) / 2.0f;
    } else if (direction == "left") {
        // 起点：左移到相邻完整色块 [neighborLeft, left]
        if (idx < 0) {
            throw std::runtime_error("起点左侧不存在相邻色块，无法校正");
        }
        float neighborLeft = (idx >= 1) ? boundaries[idx - 1] : 0.0f;
        centerX = (neighborLeft + left) / 2.0f;
    } else {
        // 终点：右移到相邻完整色块 [right, neighborRight]
        int nb = static_cast<int>(boundaries.size());
        float neighborRight;
        if (idx == -1) {
            if (nb < 2) {
                throw std::runtime_error("终点右侧不存在相邻色块，无法校正");
            }
            neighborRight = boundaries[1];
        } else if (idx + 2 <= nb - 1) {
            neighborRight = boundaries[idx + 2];
        } else {
            throw std::runtime_error("终点右侧不存在相邻色块，无法校正");
        }
        centerX = (right + neighborRight) / 2.0f;
    }

    return centerX;
}

/**
 * 重新计算并调整色标条起点和终点坐标。
 *
 * 注意：色标条右侧为起点坐标，左侧为终点坐标。
 *
 * @param imgPath 图像路径。
 * @param point_start 起点坐标（色标条右侧）。
 * @param point_end 终点坐标（色标条左侧）。
 * @param processed_point_start 输出调整后的起点坐标。
 * @param processed_point_end 输出调整后的终点坐标。
 * @param debug 是否输出 debug 图片，默认 false。
 * @param debugOutputDir debug 图片输出目录，默认 "./debug"。
 *
 * @note 校正规则：起点/终点所在色块完整时返回其中心；为半块时抛弃之，
 *       内移到相邻完整色块并返回其中心。
 * @throws std::runtime_error 分类为 unknown 时抛出；抛出前对应输出坐标
 *         已回退为原始输入坐标。
 */
void recaluate_coordinates(
    std::string imgPath,
    PointP point_start,
    PointP point_end,
    PointP& processed_point_start,
    PointP& processed_point_end,
    bool debug = false,
    std::string debugOutputDir = "./debug"
) {
    cv::Mat image = cv::imread(imgPath, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("无法读取图像: " + imgPath);
    }

    // OpenCV 默认读取 BGR，转换为 RGB
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    image.convertTo(image, CV_32F);

    int scanRangePx = DEFAULT_SCAN_RANGE_PX;
    int rowSpan = DEFAULT_ROW_SPAN;
    float edgeThreshold = DEFAULT_EDGE_THRESHOLD;
    int clusterGap = DEFAULT_CLUSTER_GAP;

    // 右侧为起点，向左扫描；左侧为终点，向右扫描
    float ratioStart = 0.0f, ratioEnd = 0.0f;
    std::string catStart, catEnd;

    // unknown 时 classifyAndAdjust 抛异常：捕获后将输出坐标回退为原始坐标，再向上抛出
    try {
        float adjustedStartX = classifyAndAdjust(
            image,
            point_start.x,
            point_start.y,
            "left",       // 起点在右侧，向左找参考色块
            scanRangePx,
            rowSpan,
            edgeThreshold,
            clusterGap,
            debug,
            "start",
            debugOutputDir,
            ratioStart,
            catStart
        );
        processed_point_start = {
            static_cast<int>(std::round(adjustedStartX)),
            point_start.y
        };
    } catch (const std::exception& e) {
        processed_point_start = point_start;
        std::cerr << "错误: 起点坐标校正失败，已回退为原始坐标: " << e.what() << std::endl;
        throw;
    }

    try {
        float adjustedEndX = classifyAndAdjust(
            image,
            point_end.x,
            point_end.y,
            "right",      // 终点在左侧，向右找参考色块
            scanRangePx,
            rowSpan,
            edgeThreshold,
            clusterGap,
            debug,
            "end",
            debugOutputDir,
            ratioEnd,
            catEnd
        );
        processed_point_end = {
            static_cast<int>(std::round(adjustedEndX)),
            point_end.y
        };
    } catch (const std::exception& e) {
        processed_point_end = point_end;
        std::cerr << "错误: 终点坐标校正失败，已回退为原始坐标: " << e.what() << std::endl;
        throw;
    }

    std::cout << "起点分类: " << catStart << ", 比例: " << ratioStart << std::endl;
    std::cout << "终点分类: " << catEnd << ", 比例: " << ratioEnd << std::endl;
}

int main() {
    try {
        std::string imgPath = "test.bmp";

        // 注意：右侧为起点，左侧为终点
        PointP start = {13983, 2195}; // 色标条右侧起点
        PointP end = {42, 2195};      // 色标条左侧终点

        PointP processedStart, processedEnd;

        // 开启 debug 模式，输出调试图片到 ./debug 目录
        recaluate_coordinates(imgPath, start, end, processedStart, processedEnd, true, "./debug");

        std::cout << "调整后起点: (" << processedStart.x << ", " << processedStart.y << ")" << std::endl;
        std::cout << "调整后终点: (" << processedEnd.x << ", " << processedEnd.y << ")" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
