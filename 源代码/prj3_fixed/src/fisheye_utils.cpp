#include "fisheye_utils.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>

namespace {

constexpr double kPi = 3.14159265358979323846;

bool isFinitePositive(double value) {
    return std::isfinite(value) && value > 0.0;
}

std::string lowerCopy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

bool hasFileStorageExtension(const std::string& path) {
    const std::string lower = lowerCopy(path);
    return lower.size() >= 4 &&
           (lower.find(".xml") != std::string::npos ||
            lower.find(".yml") != std::string::npos ||
            lower.find(".yaml") != std::string::npos);
}

bool hasNpyExtension(const std::string& path) {
    const std::string lower = lowerCopy(path);
    return lower.size() >= 4 && lower.substr(lower.size() - 4) == ".npy";
}

bool loadNpyNormal(const std::string& path, cv::Mat& normal_xyz, std::string& error_message) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        error_message = "无法打开 NPY GT 法向量文件: " + path;
        return false;
    }

    char magic[6] = {};
    input.read(magic, 6);
    if (!input || std::string(magic, 6) != "\x93NUMPY") {
        error_message = "不是有效的 NPY 文件: " + path;
        return false;
    }

    unsigned char major = 0;
    unsigned char minor = 0;
    input.read(reinterpret_cast<char*>(&major), 1);
    input.read(reinterpret_cast<char*>(&minor), 1);
    (void)minor;

    std::uint32_t header_length = 0;
    if (major == 1) {
        std::uint16_t length16 = 0;
        input.read(reinterpret_cast<char*>(&length16), 2);
        header_length = length16;
    } else if (major == 2 || major == 3) {
        input.read(reinterpret_cast<char*>(&header_length), 4);
    } else {
        error_message = "不支持的 NPY 版本: " + path;
        return false;
    }

    std::string header(header_length, '\0');
    input.read(header.data(), static_cast<std::streamsize>(header.size()));
    if (!input) {
        error_message = "读取 NPY 头失败: " + path;
        return false;
    }

    const bool float32 = header.find("'descr': '<f4'") != std::string::npos ||
                         header.find("\"descr\": \"<f4\"") != std::string::npos ||
                         header.find("'descr': '|f4'") != std::string::npos;
    const bool float64 = header.find("'descr': '<f8'") != std::string::npos ||
                         header.find("\"descr\": \"<f8\"") != std::string::npos;
    if (!float32 && !float64) {
        error_message = "NPY GT 仅支持 float32/float64: " + path;
        return false;
    }
    if (header.find("True") != std::string::npos) {
        error_message = "NPY GT 不支持 Fortran-order 数组: " + path;
        return false;
    }

    const std::size_t open = header.find('(');
    const std::size_t close = header.find(')', open);
    if (open == std::string::npos || close == std::string::npos) {
        error_message = "NPY shape 解析失败: " + path;
        return false;
    }

    std::string shape_text = header.substr(open + 1, close - open - 1);
    std::replace(shape_text.begin(), shape_text.end(), ',', ' ');
    std::istringstream shape_stream(shape_text);
    int rows = 0;
    int cols = 0;
    int channels = 0;
    shape_stream >> rows >> cols >> channels;
    if (rows <= 0 || cols <= 0 || channels != 3) {
        error_message = "NPY GT shape 必须为 HxWx3: " + path;
        return false;
    }

    normal_xyz = cv::Mat::zeros(rows, cols, CV_32FC3);
    const std::size_t total_values = static_cast<std::size_t>(rows) * cols * channels;
    if (float32) {
        input.read(reinterpret_cast<char*>(normal_xyz.data),
                   static_cast<std::streamsize>(total_values * sizeof(float)));
    } else {
        std::vector<double> buffer(total_values);
        input.read(reinterpret_cast<char*>(buffer.data()),
                   static_cast<std::streamsize>(total_values * sizeof(double)));
        for (int v = 0; v < rows; ++v) {
            for (int u = 0; u < cols; ++u) {
                cv::Vec3f& n = normal_xyz.at<cv::Vec3f>(v, u);
                const std::size_t offset = (static_cast<std::size_t>(v) * cols + u) * 3;
                n[0] = static_cast<float>(buffer[offset + 0]);
                n[1] = static_cast<float>(buffer[offset + 1]);
                n[2] = static_cast<float>(buffer[offset + 2]);
            }
        }
    }
    if (!input) {
        error_message = "读取 NPY 数据失败: " + path;
        return false;
    }
    return true;
}

bool normalizeVector(const cv::Vec3f& input, cv::Vec3d& output) {
    const double x = input[0];
    const double y = input[1];
    const double z = input[2];
    const double norm = std::sqrt(x * x + y * y + z * z);
    if (!std::isfinite(norm) || norm < 1e-8) {
        return false;
    }
    output = cv::Vec3d(x / norm, y / norm, z / norm);
    return true;
}

}  // namespace

bool backProject(int u, int v, double depth, const CameraParams& cam, cv::Point3d& point) {
    if (!isFinitePositive(depth) || !std::isfinite(cam.f) || cam.f <= 0.0) {
        return false;
    }

    const double x = static_cast<double>(u) - cam.cx;
    const double y = static_cast<double>(v) - cam.cy;
    const double r = std::hypot(x, y);

    if (cam.valid_radius > 0.0 && r > cam.valid_radius) {
        return false;
    }

    const double theta = r / cam.f;
    const double max_theta = cam.max_theta_deg * kPi / 180.0;
    if (!std::isfinite(theta) || theta > max_theta) {
        return false;
    }

    if (r < 1e-12) {
        point = cv::Point3d(0.0, 0.0, depth);
        return true;
    }

    const double cos_phi = x / r;
    const double sin_phi = y / r;
    const double sin_theta = std::sin(theta);
    const double cos_theta = std::cos(theta);

    if (cam.depth_mode == DepthMode::RayDistance) {
        point = cv::Point3d(
            depth * sin_theta * cos_phi,
            depth * sin_theta * sin_phi,
            depth * cos_theta);
        return std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z);
    }

    // Z-depth 只在射线位于相机前方、cos(theta)>0 时有稳定物理意义。
    if (cos_theta <= 1e-8) {
        return false;
    }

    const double radial_xy = depth * std::tan(theta);
    point = cv::Point3d(
        radial_xy * cos_phi,
        radial_xy * sin_phi,
        depth);
    return std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z);
}

bool fitPlaneNormalPCA(const std::vector<cv::Point3d>& points, cv::Point3d& normal) {
    if (points.size() < 3) {
        return false;
    }

    cv::Point3d centroid(0.0, 0.0, 0.0);
    for (const auto& point : points) {
        centroid += point;
    }
    centroid *= 1.0 / static_cast<double>(points.size());

    cv::Mat covariance = cv::Mat::zeros(3, 3, CV_64F);
    for (const auto& point : points) {
        const cv::Vec3d d(point.x - centroid.x, point.y - centroid.y, point.z - centroid.z);
        for (int row = 0; row < 3; ++row) {
            for (int col = row; col < 3; ++col) {
                covariance.at<double>(row, col) += d[row] * d[col];
            }
        }
    }

    covariance.at<double>(1, 0) = covariance.at<double>(0, 1);
    covariance.at<double>(2, 0) = covariance.at<double>(0, 2);
    covariance.at<double>(2, 1) = covariance.at<double>(1, 2);
    covariance /= static_cast<double>(points.size());

    cv::Mat eigenvalues;
    cv::Mat eigenvectors;
    if (!cv::eigen(covariance, eigenvalues, eigenvectors)) {
        return false;
    }

    // cv::eigen 按特征值从大到小排列；最小特征值对应平面法向量。
    const double second_eigenvalue = eigenvalues.at<double>(1, 0);
    if (!std::isfinite(second_eigenvalue) || second_eigenvalue < 1e-14) {
        // 邻域退化成一个点或一条线，无法确定唯一平面。
        return false;
    }

    normal = cv::Point3d(
        eigenvectors.at<double>(2, 0),
        eigenvectors.at<double>(2, 1),
        eigenvectors.at<double>(2, 2));

    const double norm = cv::norm(normal);
    if (!std::isfinite(norm) || norm < 1e-12) {
        return false;
    }
    normal *= 1.0 / norm;
    return true;
}

void computeNormalMap(
    const cv::Mat& depth,
    const CameraParams& cam,
    const NormalEstimationParams& params,
    cv::Mat& normal_xyz,
    cv::Mat& valid_mask) {

    CV_Assert(depth.type() == CV_32F);

    normal_xyz = cv::Mat::zeros(depth.size(), CV_32FC3);
    valid_mask = cv::Mat::zeros(depth.size(), CV_8U);

    cv::Mat point_cloud = cv::Mat::zeros(depth.size(), CV_64FC3);
    cv::Mat geometry_mask = cv::Mat::zeros(depth.size(), CV_8U);

    for (int v = 0; v < depth.rows; ++v) {
        for (int u = 0; u < depth.cols; ++u) {
            const float d = depth.at<float>(v, u);
            cv::Point3d point;
            if (backProject(u, v, static_cast<double>(d), cam, point)) {
                point_cloud.at<cv::Vec3d>(v, u) = cv::Vec3d(point.x, point.y, point.z);
                geometry_mask.at<unsigned char>(v, u) = 255;
            }
        }
    }

    const int min_window = std::max(1, params.min_window);
    const int max_window = std::max(min_window, params.max_window);
    const double growth_radius = std::max(1.0, params.window_growth_radius);

    for (int v = 0; v < depth.rows; ++v) {
        for (int u = 0; u < depth.cols; ++u) {
            if (geometry_mask.at<unsigned char>(v, u) == 0) {
                continue;
            }

            const float center_depth = depth.at<float>(v, u);
            if (!isFinitePositive(center_depth)) {
                continue;
            }

            const double r = std::hypot(static_cast<double>(u) - cam.cx,
                                        static_cast<double>(v) - cam.cy);
            int window = min_window + static_cast<int>(r / growth_radius);
            window = std::max(min_window, std::min(window, max_window));

            const double depth_threshold = std::max(
                params.min_depth_difference,
                params.discontinuity_ratio * static_cast<double>(center_depth));

            std::vector<cv::Point3d> neighbors;
            neighbors.reserve(static_cast<std::size_t>((2 * window + 1) * (2 * window + 1)));

            const int v_begin = std::max(0, v - window);
            const int v_end = std::min(depth.rows - 1, v + window);
            const int u_begin = std::max(0, u - window);
            const int u_end = std::min(depth.cols - 1, u + window);

            for (int nv = v_begin; nv <= v_end; ++nv) {
                for (int nu = u_begin; nu <= u_end; ++nu) {
                    if (geometry_mask.at<unsigned char>(nv, nu) == 0) {
                        continue;
                    }
                    const float neighbor_depth = depth.at<float>(nv, nu);
                    if (!isFinitePositive(neighbor_depth) ||
                        std::abs(static_cast<double>(neighbor_depth - center_depth)) > depth_threshold) {
                        continue;
                    }
                    const cv::Vec3d point = point_cloud.at<cv::Vec3d>(nv, nu);
                    neighbors.emplace_back(point[0], point[1], point[2]);
                }
            }

            if (neighbors.size() < params.min_neighbors) {
                continue;
            }

            cv::Point3d normal;
            if (!fitPlaneNormalPCA(neighbors, normal)) {
                continue;
            }

            const cv::Vec3d center = point_cloud.at<cv::Vec3d>(v, u);
            const double facing = normal.x * center[0] + normal.y * center[1] + normal.z * center[2];
            if (cam.normal_orientation == NormalOrientation::TowardCamera && facing > 0.0) {
                normal *= -1.0;
            } else if (cam.normal_orientation == NormalOrientation::AwayFromCamera && facing < 0.0) {
                normal *= -1.0;
            }

            normal_xyz.at<cv::Vec3f>(v, u) = cv::Vec3f(
                static_cast<float>(normal.x),
                static_cast<float>(normal.y),
                static_cast<float>(normal.z));
            valid_mask.at<unsigned char>(v, u) = 255;
        }
    }
}

cv::Mat visualizeNormals(const cv::Mat& normal_xyz, const cv::Mat& valid_mask) {
    CV_Assert(normal_xyz.type() == CV_32FC3);
    CV_Assert(valid_mask.type() == CV_8U);
    CV_Assert(normal_xyz.size() == valid_mask.size());

    cv::Mat visualization = cv::Mat::zeros(normal_xyz.size(), CV_8UC3);
    for (int v = 0; v < normal_xyz.rows; ++v) {
        for (int u = 0; u < normal_xyz.cols; ++u) {
            if (valid_mask.at<unsigned char>(v, u) == 0) {
                continue;
            }
            const cv::Vec3f n = normal_xyz.at<cv::Vec3f>(v, u);
            const auto map_component = [](float value) -> unsigned char {
                const float mapped = std::max(0.0f, std::min(255.0f, (value * 0.5f + 0.5f) * 255.0f));
                return static_cast<unsigned char>(std::lround(mapped));
            };

            // 希望保存后显示的 RGB 为 (nx, ny, nz)，因此 OpenCV BGR 写入顺序是 (nz, ny, nx)。
            visualization.at<cv::Vec3b>(v, u) = cv::Vec3b(
                map_component(n[2]),
                map_component(n[1]),
                map_component(n[0]));
        }
    }
    return visualization;
}

cv::Mat visualizeDepth(const cv::Mat& depth) {
    CV_Assert(depth.type() == CV_32F);
    cv::Mat valid_mask = cv::Mat::zeros(depth.size(), CV_8U);
    for (int v = 0; v < depth.rows; ++v) {
        for (int u = 0; u < depth.cols; ++u) {
            const float value = depth.at<float>(v, u);
            if (std::isfinite(value) && value > 0.0f) {
                valid_mask.at<unsigned char>(v, u) = 255;
            }
        }
    }

    double min_value = 0.0;
    double max_value = 0.0;
    cv::minMaxLoc(depth, &min_value, &max_value, nullptr, nullptr, valid_mask);

    cv::Mat visualization = cv::Mat::zeros(depth.size(), CV_8U);
    if (max_value <= min_value) {
        return visualization;
    }
    depth.convertTo(visualization, CV_8U, 255.0 / (max_value - min_value),
                    -min_value * 255.0 / (max_value - min_value));
    visualization.setTo(0, valid_mask == 0);
    return visualization;
}

bool loadDepthImage(const std::string& path, double depth_scale, cv::Mat& depth, std::string& error_message) {
    cv::Mat raw = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (raw.empty()) {
        error_message = "无法读取深度图: " + path;
        return false;
    }
    if (raw.channels() != 1) {
        if (raw.channels() == 3 || raw.channels() == 4) {
            std::vector<cv::Mat> channels;
            cv::split(raw, channels);
            raw = channels[0];
        } else {
            error_message = "深度图必须是单通道或同值多通道，当前通道数为 " +
                            std::to_string(raw.channels()) + ": " + path;
            return false;
        }
    }

    double scale = depth_scale;
    if (scale < 0.0) {
        scale = (raw.depth() == CV_16U || raw.depth() == CV_16S) ? 0.001 : 1.0;
    }
    raw.convertTo(depth, CV_32F, scale);

    for (int v = 0; v < depth.rows; ++v) {
        for (int u = 0; u < depth.cols; ++u) {
            float& value = depth.at<float>(v, u);
            if (!std::isfinite(value) || value <= 0.0f || value >= 1e9f) {
                value = 0.0f;
            }
        }
    }
    return true;
}

bool loadGroundTruthNormals(
    const std::string& path,
    const std::string& format,
    cv::Mat& gt_normal_xyz,
    cv::Mat& gt_valid_mask,
    std::string& error_message) {

    const std::string requested_format = lowerCopy(format);
    bool read_as_xyz = requested_format == "xyz";
    bool read_as_rgb = requested_format == "rgb";

    if (hasNpyExtension(path)) {
        if (!loadNpyNormal(path, gt_normal_xyz, error_message)) {
            return false;
        }
        read_as_xyz = true;
        read_as_rgb = false;
    } else if (hasFileStorageExtension(path)) {
        cv::FileStorage storage(path, cv::FileStorage::READ);
        if (!storage.isOpened()) {
            error_message = "无法打开 GT 法向量文件: " + path;
            return false;
        }
        cv::FileNode node = storage["normal_xyz"];
        if (node.empty()) {
            node = storage["normal"];
        }
        if (node.empty()) {
            error_message = "GT 文件中未找到 normal_xyz 或 normal 节点: " + path;
            return false;
        }
        node >> gt_normal_xyz;
        storage["valid_mask"] >> gt_valid_mask;
        storage.release();

        if (gt_normal_xyz.empty() || gt_normal_xyz.channels() != 3) {
            error_message = "GT 法向量矩阵必须为三通道: " + path;
            return false;
        }
        gt_normal_xyz.convertTo(gt_normal_xyz, CV_32FC3);
        read_as_xyz = true;
        read_as_rgb = false;
    } else {
        cv::Mat raw = cv::imread(path, cv::IMREAD_UNCHANGED);
        if (raw.empty()) {
            error_message = "无法读取 GT 法向量图: " + path;
            return false;
        }
        if (raw.channels() != 3) {
            error_message = "GT 法向量图必须是三通道: " + path;
            return false;
        }

        if (requested_format == "auto") {
            read_as_rgb = raw.depth() == CV_8U || raw.depth() == CV_16U;
            read_as_xyz = !read_as_rgb;
        }

        if (read_as_rgb) {
            const double max_value = raw.depth() == CV_16U ? 65535.0 : 255.0;
            gt_normal_xyz = cv::Mat::zeros(raw.size(), CV_32FC3);
            gt_valid_mask = cv::Mat::zeros(raw.size(), CV_8U);
            for (int v = 0; v < raw.rows; ++v) {
                for (int u = 0; u < raw.cols; ++u) {
                    double b = 0.0, g = 0.0, r = 0.0;
                    if (raw.depth() == CV_16U) {
                        const cv::Vec<unsigned short, 3> value = raw.at<cv::Vec<unsigned short, 3>>(v, u);
                        b = value[0]; g = value[1]; r = value[2];
                    } else if (raw.depth() == CV_8U) {
                        const cv::Vec3b value = raw.at<cv::Vec3b>(v, u);
                        b = value[0]; g = value[1]; r = value[2];
                    } else {
                        error_message = "rgb 格式仅支持 8/16 位整数法向量图: " + path;
                        return false;
                    }
                    gt_normal_xyz.at<cv::Vec3f>(v, u) = cv::Vec3f(
                        static_cast<float>(r / max_value * 2.0 - 1.0),
                        static_cast<float>(g / max_value * 2.0 - 1.0),
                        static_cast<float>(b / max_value * 2.0 - 1.0));
                    // 常见数据集用纯黑表示无效区域；避免将其误解码为 (-1,-1,-1)。
                    if (b > 0.0 || g > 0.0 || r > 0.0) {
                        gt_valid_mask.at<unsigned char>(v, u) = 255;
                    }
                }
            }
        } else if (read_as_xyz) {
            raw.convertTo(gt_normal_xyz, CV_32FC3);
        } else {
            error_message = "未知 GT 格式，请使用 auto、rgb 或 xyz";
            return false;
        }
    }

    if (gt_valid_mask.empty()) {
        gt_valid_mask = cv::Mat::zeros(gt_normal_xyz.size(), CV_8U);
        for (int v = 0; v < gt_normal_xyz.rows; ++v) {
            for (int u = 0; u < gt_normal_xyz.cols; ++u) {
                cv::Vec3d normalized;
                if (normalizeVector(gt_normal_xyz.at<cv::Vec3f>(v, u), normalized)) {
                    gt_valid_mask.at<unsigned char>(v, u) = 255;
                }
            }
        }
    } else {
        if (gt_valid_mask.size() != gt_normal_xyz.size()) {
            error_message = "GT valid_mask 与法向量尺寸不一致";
            return false;
        }
        if (gt_valid_mask.channels() != 1) {
            cv::cvtColor(gt_valid_mask, gt_valid_mask, cv::COLOR_BGR2GRAY);
        }
        gt_valid_mask.convertTo(gt_valid_mask, CV_8U);
        cv::threshold(gt_valid_mask, gt_valid_mask, 0, 255, cv::THRESH_BINARY);
    }

    return true;
}

EvaluationResult evaluateNormals(
    const cv::Mat& prediction_xyz,
    const cv::Mat& prediction_mask,
    const cv::Mat& gt_xyz,
    const cv::Mat& gt_mask,
    const cv::Mat& extra_mask,
    bool ignore_sign,
    cv::Mat& angular_error_deg,
    cv::Mat& evaluation_mask) {

    CV_Assert(prediction_xyz.type() == CV_32FC3 && gt_xyz.type() == CV_32FC3);
    CV_Assert(prediction_mask.type() == CV_8U && gt_mask.type() == CV_8U);
    CV_Assert(prediction_xyz.size() == gt_xyz.size());

    angular_error_deg = cv::Mat::zeros(prediction_xyz.size(), CV_32F);
    evaluation_mask = cv::Mat::zeros(prediction_xyz.size(), CV_8U);

    std::vector<double> errors;
    errors.reserve(prediction_xyz.total());

    for (int v = 0; v < prediction_xyz.rows; ++v) {
        for (int u = 0; u < prediction_xyz.cols; ++u) {
            if (prediction_mask.at<unsigned char>(v, u) == 0 ||
                gt_mask.at<unsigned char>(v, u) == 0 ||
                (!extra_mask.empty() && extra_mask.at<unsigned char>(v, u) == 0)) {
                continue;
            }

            cv::Vec3d predicted;
            cv::Vec3d target;
            if (!normalizeVector(prediction_xyz.at<cv::Vec3f>(v, u), predicted) ||
                !normalizeVector(gt_xyz.at<cv::Vec3f>(v, u), target)) {
                continue;
            }

            double dot = predicted.dot(target);
            if (ignore_sign) {
                dot = std::abs(dot);
            }
            dot = std::max(-1.0, std::min(1.0, dot));
            const double angle = std::acos(dot) * 180.0 / kPi;

            angular_error_deg.at<float>(v, u) = static_cast<float>(angle);
            evaluation_mask.at<unsigned char>(v, u) = 255;
            errors.push_back(angle);
        }
    }

    EvaluationResult result;
    result.valid_count = errors.size();
    if (errors.empty()) {
        return result;
    }

    const double sum = std::accumulate(errors.begin(), errors.end(), 0.0);
    double squared_sum = 0.0;
    std::size_t count_11_25 = 0;
    std::size_t count_22_5 = 0;
    std::size_t count_30 = 0;
    for (const double error : errors) {
        squared_sum += error * error;
        count_11_25 += error < 11.25;
        count_22_5 += error < 22.5;
        count_30 += error < 30.0;
    }

    result.mean_deg = sum / static_cast<double>(errors.size());
    result.rmse_deg = std::sqrt(squared_sum / static_cast<double>(errors.size()));
    result.pct_11_25 = 100.0 * static_cast<double>(count_11_25) / errors.size();
    result.pct_22_5 = 100.0 * static_cast<double>(count_22_5) / errors.size();
    result.pct_30 = 100.0 * static_cast<double>(count_30) / errors.size();

    const std::size_t middle = errors.size() / 2;
    std::nth_element(errors.begin(), errors.begin() + middle, errors.end());
    result.median_deg = errors[middle];
    if (errors.size() % 2 == 0) {
        const auto lower_max = std::max_element(errors.begin(), errors.begin() + middle);
        result.median_deg = (*lower_max + result.median_deg) * 0.5;
    }
    return result;
}

cv::Mat visualizeAngularError(
    const cv::Mat& angular_error_deg,
    const cv::Mat& evaluation_mask,
    double max_display_error_deg) {

    CV_Assert(angular_error_deg.type() == CV_32F);
    CV_Assert(evaluation_mask.type() == CV_8U);

    const double cap = std::max(1e-6, max_display_error_deg);
    cv::Mat normalized = cv::Mat::zeros(angular_error_deg.size(), CV_8U);
    for (int v = 0; v < angular_error_deg.rows; ++v) {
        for (int u = 0; u < angular_error_deg.cols; ++u) {
            if (evaluation_mask.at<unsigned char>(v, u) == 0) {
                continue;
            }
            const double error = std::min(cap, std::max(0.0, static_cast<double>(angular_error_deg.at<float>(v, u))));
            normalized.at<unsigned char>(v, u) = static_cast<unsigned char>(std::lround(error / cap * 255.0));
        }
    }

    cv::Mat heatmap;
    cv::applyColorMap(normalized, heatmap, cv::COLORMAP_JET);
    heatmap.setTo(cv::Scalar(0, 0, 0), evaluation_mask == 0);
    return heatmap;
}

bool saveNormalFloat(
    const std::string& path,
    const cv::Mat& normal_xyz,
    const cv::Mat& valid_mask) {

    cv::FileStorage storage(path, cv::FileStorage::WRITE);
    if (!storage.isOpened()) {
        return false;
    }
    storage << "normal_xyz" << normal_xyz;
    storage << "valid_mask" << valid_mask;
    storage.release();
    return true;
}
