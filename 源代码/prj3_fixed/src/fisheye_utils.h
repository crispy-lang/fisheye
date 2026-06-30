#pragma once

#include <opencv2/core.hpp>
#include <cstddef>
#include <string>
#include <vector>

enum class DepthMode {
    ZDepth,      // 深度值是相机坐标系中的 Z 坐标
    RayDistance  // 深度值是相机中心沿成像射线到三维点的欧氏距离
};

enum class NormalOrientation {
    None,
    TowardCamera,
    AwayFromCamera
};

struct CameraParams {
    double f = 250.0;              // 等距投影焦距：r = f * theta
    double cx = -1.0;              // 光心；小于 0 时自动取图像中心
    double cy = -1.0;
    double max_theta_deg = 89.0;   // 最大有效入射角（从光轴量起）
    double valid_radius = -1.0;    // 有效鱼眼圆半径；<=0 表示仅使用 max_theta_deg
    DepthMode depth_mode = DepthMode::ZDepth;
    NormalOrientation normal_orientation = NormalOrientation::TowardCamera;
};

struct NormalEstimationParams {
    double discontinuity_ratio = 0.08;  // 邻域深度差阈值 = ratio * 中心深度
    double min_depth_difference = 1e-4;
    int min_window = 1;
    int max_window = 3;
    double window_growth_radius = 150.0;
    std::size_t min_neighbors = 5;
};

struct EvaluationResult {
    std::size_t valid_count = 0;
    double mean_deg = 0.0;
    double median_deg = 0.0;
    double rmse_deg = 0.0;
    double pct_11_25 = 0.0;
    double pct_22_5 = 0.0;
    double pct_30 = 0.0;
};

// 像素反投影到相机坐标系。无效像素返回 false。
bool backProject(int u, int v, double depth, const CameraParams& cam, cv::Point3d& point);

// 使用 PCA / 总最小二乘拟合任意朝向平面，避免 Z=AX+BY+C 对竖直表面退化。
bool fitPlaneNormalPCA(const std::vector<cv::Point3d>& points, cv::Point3d& normal);

// 直接在原始鱼眼深度网格上估计法向量，不做重采样或透视重映射。
// normal_xyz: CV_32FC3，通道顺序为 (nx, ny, nz)。
// valid_mask: CV_8U，255 表示该像素得到有效预测。
void computeNormalMap(
    const cv::Mat& depth,
    const CameraParams& cam,
    const NormalEstimationParams& params,
    cv::Mat& normal_xyz,
    cv::Mat& valid_mask);

// 将 XYZ 法向量映射为常见 RGB=(nx,ny,nz) 的 8 位可视化图；OpenCV 内部正确处理 BGR 顺序。
cv::Mat visualizeNormals(const cv::Mat& normal_xyz, const cv::Mat& valid_mask);

// 深度图可视化，仅使用有效正深度确定归一化范围。
cv::Mat visualizeDepth(const cv::Mat& depth);

// 读取深度图并转换为 CV_32F。depth_scale < 0 时：16 位整数自动按毫米转米，浮点保持原单位。
bool loadDepthImage(const std::string& path, double depth_scale, cv::Mat& depth, std::string& error_message);

// 读取 GT 法向量。
// format=auto：整数三通道图按 RGB 编码 [0,max] -> [-1,1]；浮点图或 FileStorage 按 XYZ 浮点读取。
// format=rgb：强制按颜色编码读取；format=xyz：强制按三通道 XYZ 数值读取。
bool loadGroundTruthNormals(
    const std::string& path,
    const std::string& format,
    cv::Mat& gt_normal_xyz,
    cv::Mat& gt_valid_mask,
    std::string& error_message);

EvaluationResult evaluateNormals(
    const cv::Mat& prediction_xyz,
    const cv::Mat& prediction_mask,
    const cv::Mat& gt_xyz,
    const cv::Mat& gt_mask,
    const cv::Mat& extra_mask,
    bool ignore_sign,
    cv::Mat& angular_error_deg,
    cv::Mat& evaluation_mask);

cv::Mat visualizeAngularError(
    const cv::Mat& angular_error_deg,
    const cv::Mat& evaluation_mask,
    double max_display_error_deg);

bool saveNormalFloat(
    const std::string& path,
    const cv::Mat& normal_xyz,
    const cv::Mat& valid_mask);
