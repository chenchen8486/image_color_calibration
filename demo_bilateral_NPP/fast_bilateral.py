# -*- coding: utf-8 -*-
"""
快速双边滤波（最终方案，对应 C++ 移植流程）

流程：全分辨率 NPP 双边滤波 r=2，无降采样/上采样，避免重采样失真。
输入：input/test.bmp
输出：output/final_result.bmp （唯一结果文件）

实测（8192x4833x3, RTX 3070 Laptop）：GPU 纯计算约 8~10ms，
与 CPU OpenCV 精确双边 (cv2.bilateralFilter d=9, 25, 25) 的平均像素差 MAD ≈ 1.06。

参数说明：NPP 对彩色图每通道独立计算范围权重，OpenCV 是三通道联合权重，
因此 NPP 的 sigmaColor 需要标定补偿（经验值 ≈ OpenCV sigmaColor / 2.3）。
本方案 SIGMA_COLOR=11 即在此测试图上扫描得到的最优值；sigmaSpace 保持 25 不变
（sigmaSpace=25 相对 5x5 窗口接近平坦权重，实测对结果几乎无影响）。
已接近精度上限：即使完美复刻 OpenCV 公式的自写 kernel，MAD 也只能再降约 0.1。

依赖：本机 CUDA 工具包自带的 NPP 动态库（ctypes 直调，无需编译）。
"""
import os
import sys
import ctypes
from ctypes import c_int, c_float, c_void_p, c_size_t, byref, Structure

import numpy as np
import cv2

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_PATH = os.path.join(BASE_DIR, "input", "test.bmp")
OUT_PATH = os.path.join(BASE_DIR, "output", "final_result.bmp")

CUDA_BIN = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4\bin"
os.add_dll_directory(CUDA_BIN)
cudart = ctypes.CDLL(os.path.join(CUDA_BIN, "cudart64_110.dll"))
nppif = ctypes.CDLL(os.path.join(CUDA_BIN, "nppif64_11.dll"))

NPP_BORDER_REPLICATE = 2
H2D, D2H = 1, 2

BILATERAL_RADIUS = 2       # 5x5 窗口
SIGMA_COLOR = 11.0         # 范围权重 sigma：针对 NPP 逐通道权重标定，逼近 OpenCV 联合权重 sigmaColor=25 的效果
SIGMA_SPACE = 25.0         # 空间权重 sigma，与 cv2.bilateralFilter 的 sigmaSpace 对应


class NppiSize(Structure):
    _fields_ = [("width", c_int), ("height", c_int)]


class NppiPoint(Structure):
    _fields_ = [("x", c_int), ("y", c_int)]


bilateral = nppif.nppiFilterBilateralGaussBorder_8u_C3R
bilateral.restype = c_int
bilateral.argtypes = [c_void_p, c_int, NppiSize, NppiPoint,
                      c_void_p, c_int, NppiSize,
                      c_int, c_int, c_float, c_float, c_int]


def check(st, what):
    if st != 0:
        raise RuntimeError(f"{what} -> NppStatus {st}")


class GpuImage:
    """设备端 C3 8u 图像（pitch 对齐）"""
    def __init__(self, w, h):
        self.w, self.h, self.row_bytes = w, h, w * 3
        self.ptr, self.pitch = c_void_p(), c_size_t()
        check(cudart.cudaMallocPitch(byref(self.ptr), byref(self.pitch),
                                     self.row_bytes, h), "cudaMallocPitch")

    def free(self):
        cudart.cudaFree(self.ptr)


class Timer:
    def __init__(self):
        self.start, self.stop = c_void_p(), c_void_p()
        cudart.cudaEventCreate(byref(self.start))
        cudart.cudaEventCreate(byref(self.stop))
        self.el = c_float()

    def measure(self, fn, n=10, warmup=2):
        for _ in range(warmup):
            fn()
        cudart.cudaDeviceSynchronize()
        ts = []
        for _ in range(n):
            cudart.cudaEventRecord(self.start, None)
            fn()
            cudart.cudaEventRecord(self.stop, None)
            cudart.cudaEventSynchronize(self.stop)
            cudart.cudaEventElapsedTime(byref(self.el), self.start, self.stop)
            ts.append(self.el.value)
        return min(ts)


def main():
    img = cv2.imread(IMG_PATH)
    assert img is not None and img.flags["C_CONTIGUOUS"], f"读取失败: {IMG_PATH}"
    h, w = img.shape[:2]
    total = w * 3 * h
    print(f"输入: {IMG_PATH}  {w}x{h}x3 ({total / 1e6:.1f} MB)")

    # 锁页主机内存（达到满速 PCIe 传输）
    host_src, host_dst = c_void_p(), c_void_p()
    check(cudart.cudaHostAlloc(byref(host_src), total, 0), "cudaHostAlloc")
    check(cudart.cudaHostAlloc(byref(host_dst), total, 0), "cudaHostAlloc")
    ctypes.memmove(host_src, img.ctypes.data, total)

    full, out = GpuImage(w, h), GpuImage(w, h)
    timer = Timer()

    def h2d():
        cudart.cudaMemcpy2D(full.ptr, full.pitch.value, host_src, full.row_bytes,
                            full.row_bytes, h, H2D)

    def d2h():
        cudart.cudaMemcpy2D(host_dst, out.row_bytes, out.ptr, out.pitch.value,
                            out.row_bytes, h, D2H)

    def pipeline():
        check(bilateral(full.ptr, full.pitch.value, NppiSize(w, h), NppiPoint(0, 0),
                        out.ptr, out.pitch.value, NppiSize(w, h),
                        BILATERAL_RADIUS, 1,
                        SIGMA_COLOR * SIGMA_COLOR, SIGMA_SPACE * SIGMA_SPACE,
                        NPP_BORDER_REPLICATE),
              "bilateral")

    # ---- 计时 ----
    t_h2d = timer.measure(h2d)
    t_gpu = timer.measure(pipeline)
    h2d()  # 让 full 中有数据
    t_d2h = timer.measure(d2h)
    t_e2e = timer.measure(lambda: (h2d(), pipeline(), d2h()), n=5, warmup=1)

    # ---- 正式跑一遍并取回结果 ----
    pipeline()
    cudart.cudaDeviceSynchronize()
    d2h()
    cudart.cudaDeviceSynchronize()
    result = np.frombuffer((ctypes.c_char * total).from_address(host_dst.value),
                           dtype=np.uint8).reshape(h, w, 3).copy()

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    cv2.imwrite(OUT_PATH, result)

    # ---- 与 CPU 精确双边对比 (基准固定为 d=9, sigma 25/25) ----
    ref = cv2.bilateralFilter(img, 9, 25, 25)
    mad = float(np.abs(result.astype(np.int16) - ref.astype(np.int16)).mean())

    print(f"\nGPU 双边 (全分辨率 r={BILATERAL_RADIUS})  : {t_gpu:6.2f} ms")
    print(f"PCIe 传输 H2D / D2H                   : {t_h2d:6.2f} / {t_d2h:.2f} ms")
    print(f"端到端 (指针进 -> Mat出)               : {t_e2e:6.2f} ms")
    print(f"与 CPU 精确双边的 MAD (0-255)          : {mad:.3f}")
    print(f"\n结果已保存: {OUT_PATH}")

    full.free()
    out.free()
    cudart.cudaFreeHost(host_src)
    cudart.cudaFreeHost(host_dst)


if __name__ == "__main__":
    main()
