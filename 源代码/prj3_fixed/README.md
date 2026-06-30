# PRJ3：等距鱼眼深度图直接法向量估计（修正版）

本项目在**不对原始深度图进行透视重映射或重采样**的前提下，根据等距鱼眼模型

\[
r=f\theta
\]

将每个有效深度像素反投影到三维相机坐标系，再用局部三维点的 PCA / 总最小二乘平面拟合估计法向量。

## 1. 本次修正内容

1. 将原来的 `Z=AX+BY+C` 拟合替换为三维协方差矩阵 PCA，可处理墙面、立柱等接近竖直的表面。
2. 法向量朝向改为通过 `n·P` 判断，保证法向量朝向相机，而不是只判断 `n.z`。
3. 支持两种深度定义：
   - `z`：深度值是相机坐标系中的 Z 坐标；
   - `ray`：深度值是沿成像射线的欧氏距离。
4. 加入最大入射角和有效鱼眼圆半径检查，不再把 `cos(theta)<=0` 强行改成极小正数。
5. 修正 OpenCV BGR 写入顺序，使保存后的法向量图真正满足 `RGB=(nx,ny,nz)`。
6. 输入路径错误时直接报错退出，不再自动生成假数据并显示“运行成功”。
7. 新增显式合成平面自检：`--synthetic=true`。
8. 新增 GT 法向量读取、平均/中位/RMSE 角度误差、阈值准确率和误差热力图。
9. 新增目录批处理和 `metrics.csv` 汇总。
10. 浮点结果改为压缩的 `normal_xyz.yml.gz`，并同时保存有效预测 mask。

## 2. 环境

- CMake >= 3.10
- C++17
- OpenCV（core、imgcodecs、imgproc）

Ubuntu 示例：

```bash
sudo apt update
sudo apt install build-essential cmake libopencv-dev
```

## 3. 编译

请不要使用压缩包里其他机器生成的旧 `build`。重新编译：

```bash
cd final_lab_fixed
rm -rf build
cmake -S . -B build
cmake --build build -j
```

生成程序：

```text
build/fisheye_normal
```

Windows + Visual Studio：

```powershell
cmake -S . -B build
cmake --build build --config Release
```

程序通常位于：

```text
build/Release/fisheye_normal.exe
```

## 4. 先运行合成自检

```bash
./build/fisheye_normal \
  --synthetic=true \
  --output=output/synthetic \
  --f=250 \
  --max-theta=70
```

输出中会直接显示平均角度误差。无噪声倾斜平面的误差应接近 0°，边界处可能因邻域不足被 mask 排除。

## 5. 单张真实深度图

### 5.1 16 位毫米 Z-depth

```bash
./build/fisheye_normal \
  --input=data/depth/0001.png \
  --output=output/0001 \
  --f=420.0 \
  --cx=639.5 \
  --cy=359.5 \
  --depth-mode=z \
  --depth-scale=0.001 \
  --max-theta=88 \
  --valid-radius=620
```

### 5.2 浮点米制 ray distance

```bash
./build/fisheye_normal \
  --input=data/depth/0001.exr \
  --output=output/0001 \
  --f=420.0 \
  --cx=639.5 \
  --cy=359.5 \
  --depth-mode=ray \
  --depth-scale=1.0 \
  --max-theta=100 \
  --valid-radius=620
```

> `f、cx、cy、depth-mode、depth-scale、max-theta、valid-radius` 必须根据数据集说明填写，不能直接沿用示例数值。

## 6. 单张图像定量评测

GT 为普通 8/16 位 RGB 法向量图，颜色编码满足 `RGB=(nx,ny,nz)` 且从 `[-1,1]` 映射到整数范围：

```bash
./build/fisheye_normal \
  --input=data/depth/0001.png \
  --gt=data/normal/0001.png \
  --gt-format=rgb \
  --mask=data/mask/0001.png \
  --output=output/0001 \
  --f=420.0 --cx=639.5 --cy=359.5 \
  --depth-mode=z --depth-scale=0.001
```

GT 为三通道浮点 XYZ 或 OpenCV `yml/xml`：

```bash
--gt-format=xyz
```

若 GT 法向量方向没有统一，可使用：

```bash
--ignore-sign=true
```

正式报告中应说明是否忽略正负号，不能只写数值。

## 7. 批量数据集测试

深度目录与 GT 目录中的文件应具有相同主文件名，例如：

```text
data/depth/0001.png
data/normal/0001.png
data/mask/0001.png
```

运行：

```bash
./build/fisheye_normal \
  --input=data/depth \
  --gt=data/normal \
  --mask=data/mask \
  --gt-format=rgb \
  --output=output/dataset \
  --f=420.0 --cx=639.5 --cy=359.5 \
  --depth-mode=z --depth-scale=0.001 \
  --max-theta=88 --valid-radius=620
```

批处理输出：

```text
output/dataset/metrics.csv
output/dataset/0001/depth_vis.png
output/dataset/0001/normal_map.png
output/dataset/0001/normal_xyz.yml.gz
output/dataset/0001/valid_mask.png
output/dataset/0001/angular_error_heatmap.png
output/dataset/0001/evaluation_mask.png
output/dataset/0001/metrics.txt
```

## 8. 指标含义

程序输出：

- `mean_deg`：平均角度误差；
- `median_deg`：中位角度误差；
- `rmse_deg`：角度误差均方根；
- `pct_11_25`：误差小于 11.25° 的有效像素比例；
- `pct_22_5`：误差小于 22.5° 的有效像素比例；
- `pct_30`：误差小于 30° 的有效像素比例。

像素角度误差：

\[
e_i=\arccos\left(\operatorname{clip}(\hat{\mathbf n}_i\cdot\mathbf n_i,-1,1)\right)\frac{180}{\pi}
\]

使用 `--ignore-sign=true` 时，将点积替换为绝对值。

## 9. 常用参数

```text
--disc-ratio=0.08       深度不连续相对阈值
--min-window=1          最小邻域半径
--max-window=3          最大邻域半径
--window-growth=150     随鱼眼半径增大窗口的速度
--min-neighbors=5       PCA 最少点数
--heatmap-max=60        热力图显示上限
--recursive=true        递归遍历深度目录
--save-float=false      不保存浮点法向量
```

这些是算法超参数，建议在验证集上调节，不能用测试集 GT 反复调参。

## 10. 开始写最终报告前必须确认

- 数据集深度是 Z-depth 还是 ray distance；
- 深度单位和缩放比例；
- 等距投影的 `f、cx、cy`；
- 有效鱼眼半径或最大入射角；
- GT 法向量编码和坐标系方向；
- GT 是否已统一朝向相机；
- 是否有官方有效区域 mask；
- 批量评测得到的整体指标和代表性可视化结果。
