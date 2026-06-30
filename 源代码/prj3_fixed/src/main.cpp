#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "fisheye_utils.h"

namespace fs = std::filesystem;

namespace {

constexpr double kPi = 3.14159265358979323846;

struct ProcessResult {
    bool success = false;
    bool has_metrics = false;
    EvaluationResult metrics;
    std::string message;
};

std::string lowerCopy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

bool isDepthImageFile(const fs::path& path) {
    const std::string extension = lowerCopy(path.extension().string());
    return extension == ".png" || extension == ".tif" || extension == ".tiff" ||
           extension == ".exr" || extension == ".pfm" || extension == ".bmp";
}

bool isNormalFile(const fs::path& path) {
    const std::string lower = lowerCopy(path.filename().string());
    return isDepthImageFile(path) ||
           lower.find(".npy") != std::string::npos ||
           lower.find(".xml") != std::string::npos ||
           lower.find(".yml") != std::string::npos ||
           lower.find(".yaml") != std::string::npos;
}

std::string sampleKey(const fs::path& path) {
    fs::path raw_stem = path.stem();
    if (lowerCopy(path.extension().string()) == ".gz") {
        raw_stem = raw_stem.stem();
    }
    std::string stem = lowerCopy(raw_stem.string());
    const std::vector<std::string> prefixes = {
        "camera_l_depth_",
        "camera_l_normal_",
        "depth_",
        "normal_"
    };
    for (const std::string& prefix : prefixes) {
        if (stem.rfind(prefix, 0) == 0) {
            return stem.substr(prefix.size());
        }
    }
    return stem;
}

std::string logicalStem(const fs::path& path) {
    fs::path stem = path.stem();
    if (lowerCopy(path.extension().string()) == ".gz") {
        stem = stem.stem();
    }
    return stem.string();
}

std::vector<fs::path> collectDepthFiles(const fs::path& directory, bool recursive) {
    std::vector<fs::path> files;
    std::error_code error;
    if (recursive) {
        for (const auto& entry : fs::recursive_directory_iterator(directory, error)) {
            if (error) break;
            if (entry.is_regular_file() && isDepthImageFile(entry.path())) {
                files.push_back(entry.path());
            }
        }
    } else {
        for (const auto& entry : fs::directory_iterator(directory, error)) {
            if (error) break;
            if (entry.is_regular_file() && isDepthImageFile(entry.path())) {
                files.push_back(entry.path());
            }
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

std::optional<fs::path> findMatchingFile(const fs::path& directory, const std::string& stem, bool normal_file) {
    if (!fs::is_directory(directory)) {
        return std::nullopt;
    }
    std::error_code error;
    for (const auto& entry : fs::directory_iterator(directory, error)) {
        if (error) break;
        if (!entry.is_regular_file()) continue;
        if (normal_file ? !isNormalFile(entry.path()) : !isDepthImageFile(entry.path())) continue;
        if (logicalStem(entry.path()) == stem || sampleKey(entry.path()) == sampleKey(stem)) {
            return entry.path();
        }
    }
    return std::nullopt;
}

bool loadBinaryMask(const std::string& path, const cv::Size& expected_size, cv::Mat& mask, std::string& error) {
    if (path.empty()) {
        mask.release();
        return true;
    }
    cv::Mat raw = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (raw.empty()) {
        error = "无法读取评测 mask: " + path;
        return false;
    }
    if (raw.size() != expected_size) {
        error = "评测 mask 尺寸与深度图不一致: " + path;
        return false;
    }
    cv::threshold(raw, mask, 0, 255, cv::THRESH_BINARY);
    return true;
}

void writeMetricsText(const fs::path& path, const EvaluationResult& result) {
    std::ofstream output(path);
    output << std::fixed << std::setprecision(6);
    output << "valid_count=" << result.valid_count << '\n';
    output << "mean_angular_error_deg=" << result.mean_deg << '\n';
    output << "median_angular_error_deg=" << result.median_deg << '\n';
    output << "rmse_angular_error_deg=" << result.rmse_deg << '\n';
    output << "percent_below_11.25_deg=" << result.pct_11_25 << '\n';
    output << "percent_below_22.5_deg=" << result.pct_22_5 << '\n';
    output << "percent_below_30_deg=" << result.pct_30 << '\n';
}

ProcessResult processDepth(
    const cv::Mat& depth,
    const std::string& sample_name,
    const fs::path& output_directory,
    CameraParams camera,
    const NormalEstimationParams& estimation,
    const std::string& gt_path,
    const std::string& gt_format,
    const std::string& mask_path,
    bool ignore_sign,
    bool save_float,
    double heatmap_max,
    const cv::Mat& synthetic_gt = cv::Mat(),
    const cv::Mat& synthetic_gt_mask = cv::Mat()) {

    ProcessResult result;
    std::error_code error;
    fs::create_directories(output_directory, error);
    if (error) {
        result.message = "无法创建输出目录: " + output_directory.string();
        return result;
    }

    if (camera.cx < 0.0) camera.cx = (depth.cols - 1) * 0.5;
    if (camera.cy < 0.0) camera.cy = (depth.rows - 1) * 0.5;

    std::cout << "\n[处理] " << sample_name << '\n';
    std::cout << "  图像尺寸: " << depth.cols << " x " << depth.rows << '\n';
    std::cout << "  相机参数: f=" << camera.f << ", cx=" << camera.cx << ", cy=" << camera.cy
              << ", max_theta=" << camera.max_theta_deg << " deg\n";
    std::cout << "  深度定义: "
              << (camera.depth_mode == DepthMode::ZDepth ? "Z-depth" : "ray distance") << '\n';

    cv::Mat normal_xyz;
    cv::Mat prediction_mask;
    computeNormalMap(depth, camera, estimation, normal_xyz, prediction_mask);

    const cv::Mat normal_visualization = visualizeNormals(normal_xyz, prediction_mask);
    const cv::Mat depth_visualization = visualizeDepth(depth);
    cv::imwrite((output_directory / "normal_map.png").string(), normal_visualization);
    cv::imwrite((output_directory / "depth_vis.png").string(), depth_visualization);
    cv::imwrite((output_directory / "valid_mask.png").string(), prediction_mask);

    if (save_float) {
        if (!saveNormalFloat((output_directory / "normal_xyz.yml.gz").string(), normal_xyz, prediction_mask)) {
            std::cerr << "  [警告] 浮点法向量保存失败\n";
        }
    }

    cv::Mat gt_xyz;
    cv::Mat gt_mask;
    std::string load_error;
    bool have_gt = false;
    if (!synthetic_gt.empty()) {
        synthetic_gt.copyTo(gt_xyz);
        synthetic_gt_mask.copyTo(gt_mask);
        have_gt = true;
    } else if (!gt_path.empty()) {
        have_gt = loadGroundTruthNormals(gt_path, gt_format, gt_xyz, gt_mask, load_error);
        if (!have_gt) {
            result.message = load_error;
            return result;
        }
    }

    if (have_gt) {
        if (gt_xyz.size() != depth.size()) {
            result.message = "GT 法向量尺寸与深度图不一致";
            return result;
        }

        cv::Mat extra_mask;
        if (!loadBinaryMask(mask_path, depth.size(), extra_mask, load_error)) {
            result.message = load_error;
            return result;
        }

        cv::Mat angular_error;
        cv::Mat evaluation_mask;
        result.metrics = evaluateNormals(
            normal_xyz, prediction_mask, gt_xyz, gt_mask, extra_mask,
            ignore_sign, angular_error, evaluation_mask);
        result.has_metrics = true;

        const cv::Mat heatmap = visualizeAngularError(angular_error, evaluation_mask, heatmap_max);
        cv::imwrite((output_directory / "angular_error_heatmap.png").string(), heatmap);
        cv::imwrite((output_directory / "evaluation_mask.png").string(), evaluation_mask);
        writeMetricsText(output_directory / "metrics.txt", result.metrics);

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "  有效评测像素: " << result.metrics.valid_count << '\n';
        std::cout << "  平均角度误差: " << result.metrics.mean_deg << " deg\n";
        std::cout << "  中位角度误差: " << result.metrics.median_deg << " deg\n";
        std::cout << "  <11.25° / <22.5° / <30°: "
                  << result.metrics.pct_11_25 << "% / "
                  << result.metrics.pct_22_5 << "% / "
                  << result.metrics.pct_30 << "%\n";
    }

    result.success = true;
    result.message = "完成";
    return result;
}

void generateSyntheticPlane(
    int width,
    int height,
    CameraParams camera,
    cv::Mat& depth,
    cv::Mat& gt_normal,
    cv::Mat& gt_mask) {

    if (camera.cx < 0.0) camera.cx = (width - 1) * 0.5;
    if (camera.cy < 0.0) camera.cy = (height - 1) * 0.5;

    depth = cv::Mat::zeros(height, width, CV_32F);
    gt_normal = cv::Mat::zeros(height, width, CV_32FC3);
    gt_mask = cv::Mat::zeros(height, width, CV_8U);

    cv::Vec3d plane_normal(0.30, -0.20, -1.0);
    plane_normal /= cv::norm(plane_normal);
    const cv::Vec3d point_on_plane(0.0, 0.0, 5.0);
    const double plane_constant = plane_normal.dot(point_on_plane);
    const double max_theta = camera.max_theta_deg * kPi / 180.0;

    for (int v = 0; v < height; ++v) {
        for (int u = 0; u < width; ++u) {
            const double x = static_cast<double>(u) - camera.cx;
            const double y = static_cast<double>(v) - camera.cy;
            const double r = std::hypot(x, y);
            if (camera.valid_radius > 0.0 && r > camera.valid_radius) continue;
            const double theta = r / camera.f;
            if (theta > max_theta) continue;

            cv::Vec3d ray;
            if (r < 1e-12) {
                ray = cv::Vec3d(0.0, 0.0, 1.0);
            } else {
                ray = cv::Vec3d(
                    std::sin(theta) * x / r,
                    std::sin(theta) * y / r,
                    std::cos(theta));
            }

            const double denominator = plane_normal.dot(ray);
            if (std::abs(denominator) < 1e-9) continue;
            const double ray_distance = plane_constant / denominator;
            if (!std::isfinite(ray_distance) || ray_distance <= 0.0) continue;

            const double stored_depth = camera.depth_mode == DepthMode::RayDistance
                ? ray_distance
                : ray_distance * ray[2];
            if (!std::isfinite(stored_depth) || stored_depth <= 0.0) continue;

            depth.at<float>(v, u) = static_cast<float>(stored_depth);
            gt_normal.at<cv::Vec3f>(v, u) = cv::Vec3f(
                static_cast<float>(plane_normal[0]),
                static_cast<float>(plane_normal[1]),
                static_cast<float>(plane_normal[2]));
            gt_mask.at<unsigned char>(v, u) = 255;
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    const cv::String keys =
        "{help h usage ? |      | 显示帮助 }"
        "{input i        |      | 单张深度图路径或深度图目录 }"
        "{output o       |output| 输出目录 }"
        "{gt             |      | 单张 GT 法向量路径或 GT 目录 }"
        "{mask           |      | 单张评测 mask 路径或 mask 目录 }"
        "{gt-format      |auto  | GT 格式：auto、rgb、xyz }"
        "{f              |250.0 | 等距投影焦距（像素） }"
        "{cx             |-1.0  | 光心 x；负数表示自动取图像中心 }"
        "{cy             |-1.0  | 光心 y；负数表示自动取图像中心 }"
        "{depth-mode     |z     | 深度定义：z 或 ray }"
        "{normal-orientation|toward| 法向量朝向：toward、away 或 none }"
        "{depth-scale    |-1.0  | 深度缩放；负数表示 16U 自动 mm->m，浮点保持不变 }"
        "{max-theta      |89.0  | 最大有效入射角（度） }"
        "{valid-radius   |-1.0  | 有效鱼眼圆半径；负数表示不单独限制 }"
        "{disc-ratio     |0.08  | 深度不连续相对阈值 }"
        "{min-window     |1     | 最小邻域半径 }"
        "{max-window     |3     | 最大邻域半径 }"
        "{window-growth  |150.0 | 窗口每增大一级对应的径向像素距离 }"
        "{min-neighbors  |5     | PCA 拟合所需最少邻域点数 }"
        "{ignore-sign    |false | 评测时是否忽略法向量正负号 }"
        "{heatmap-max    |60.0  | 误差热力图显示上限（度） }"
        "{recursive      |false | 批处理时是否递归遍历子目录 }"
        "{save-float     |true  | 是否保存 normal_xyz.yml.gz }"
        "{synthetic      |false | 运行带真值的合成倾斜平面自检 }"
        "{synthetic-w    |640   | 合成测试宽度 }"
        "{synthetic-h    |480   | 合成测试高度 }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Equidistant fisheye depth-to-normal estimation and evaluation");
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }
    if (!parser.check()) {
        parser.printErrors();
        return 2;
    }

    const std::string input = parser.get<std::string>("input");
    const fs::path output_root = parser.get<std::string>("output");
    const std::string gt_argument = parser.get<std::string>("gt");
    const std::string mask_argument = parser.get<std::string>("mask");
    const std::string gt_format = lowerCopy(parser.get<std::string>("gt-format"));
    const double depth_scale = parser.get<double>("depth-scale");
    const bool ignore_sign = parser.get<bool>("ignore-sign");
    const bool recursive = parser.get<bool>("recursive");
    const bool save_float = parser.get<bool>("save-float");
    const bool synthetic = parser.get<bool>("synthetic");
    const double heatmap_max = parser.get<double>("heatmap-max");

    CameraParams camera;
    camera.f = parser.get<double>("f");
    camera.cx = parser.get<double>("cx");
    camera.cy = parser.get<double>("cy");
    camera.max_theta_deg = parser.get<double>("max-theta");
    camera.valid_radius = parser.get<double>("valid-radius");
    const std::string depth_mode = lowerCopy(parser.get<std::string>("depth-mode"));
    if (depth_mode == "z" || depth_mode == "zdepth" || depth_mode == "z-depth") {
        camera.depth_mode = DepthMode::ZDepth;
    } else if (depth_mode == "ray" || depth_mode == "distance" || depth_mode == "ray-distance") {
        camera.depth_mode = DepthMode::RayDistance;
    } else {
        std::cerr << "未知 depth-mode：" << depth_mode << "，请使用 z 或 ray\n";
        return 2;
    }
    const std::string normal_orientation = lowerCopy(parser.get<std::string>("normal-orientation"));
    if (normal_orientation == "toward" || normal_orientation == "towards" ||
        normal_orientation == "camera" || normal_orientation == "toward-camera") {
        camera.normal_orientation = NormalOrientation::TowardCamera;
    } else if (normal_orientation == "away" || normal_orientation == "away-camera" ||
               normal_orientation == "away-from-camera") {
        camera.normal_orientation = NormalOrientation::AwayFromCamera;
    } else if (normal_orientation == "none" || normal_orientation == "raw") {
        camera.normal_orientation = NormalOrientation::None;
    } else {
        std::cerr << "未知 normal-orientation：" << normal_orientation
                  << "，请使用 toward、away 或 none\n";
        return 2;
    }

    NormalEstimationParams estimation;
    estimation.discontinuity_ratio = parser.get<double>("disc-ratio");
    estimation.min_window = parser.get<int>("min-window");
    estimation.max_window = parser.get<int>("max-window");
    estimation.window_growth_radius = parser.get<double>("window-growth");
    estimation.min_neighbors = static_cast<std::size_t>(std::max(3, parser.get<int>("min-neighbors")));

    std::cout << "============================================================\n";
    std::cout << " 等距鱼眼深度图直接法向量估计（PCA 修正版 + 数据集评测）\n";
    std::cout << "============================================================\n";

    if (synthetic) {
        cv::Mat depth;
        cv::Mat gt_normal;
        cv::Mat gt_mask;
        generateSyntheticPlane(
            parser.get<int>("synthetic-w"), parser.get<int>("synthetic-h"),
            camera, depth, gt_normal, gt_mask);
        const ProcessResult result = processDepth(
            depth, "synthetic_plane", output_root, camera, estimation,
            "", "xyz", "", false, save_float, heatmap_max,
            gt_normal, gt_mask);
        if (!result.success) {
            std::cerr << "[失败] " << result.message << '\n';
            return 1;
        }
        std::cout << "\n合成自检完成，结果位于: " << output_root << '\n';
        return 0;
    }

    if (input.empty()) {
        std::cerr << "必须使用 --input 指定深度图/目录，或使用 --synthetic=true 运行自检。\n";
        parser.printMessage();
        return 2;
    }

    const fs::path input_path(input);
    if (!fs::exists(input_path)) {
        std::cerr << "输入路径不存在: " << input << '\n';
        return 1;
    }

    std::vector<fs::path> depth_files;
    const bool batch_mode = fs::is_directory(input_path);
    if (batch_mode) {
        depth_files = collectDepthFiles(input_path, recursive);
    } else if (isDepthImageFile(input_path)) {
        depth_files.push_back(input_path);
    }

    if (depth_files.empty()) {
        std::cerr << "未找到可读取的深度图。支持 png/tif/tiff/exr/pfm/bmp。\n";
        return 1;
    }

    std::error_code create_error;
    fs::create_directories(output_root, create_error);
    if (create_error) {
        std::cerr << "无法创建输出目录: " << output_root << '\n';
        return 1;
    }

    std::ofstream csv;
    if (batch_mode) {
        csv.open(output_root / "metrics.csv");
        csv << "sample,status,valid_count,mean_deg,median_deg,rmse_deg,pct_11_25,pct_22_5,pct_30,message\n";
    }

    std::size_t success_count = 0;
    std::size_t evaluated_count = 0;
    double weighted_angle_sum = 0.0;
    std::size_t total_valid_pixels = 0;

    for (const fs::path& depth_path : depth_files) {
        cv::Mat depth;
        std::string load_error;
        if (!loadDepthImage(depth_path.string(), depth_scale, depth, load_error)) {
            std::cerr << "[失败] " << load_error << '\n';
            if (csv.is_open()) {
                csv << '"' << logicalStem(depth_path) << "\",failed,0,0,0,0,0,0,0,\"" << load_error << "\"\n";
            }
            continue;
        }

        const std::string stem = logicalStem(depth_path);
        const fs::path sample_output = batch_mode ? output_root / stem : output_root;

        std::string gt_path;
        if (!gt_argument.empty()) {
            const fs::path gt_base(gt_argument);
            if (fs::is_directory(gt_base)) {
                const auto match = findMatchingFile(gt_base, stem, true);
                if (match.has_value()) gt_path = match->string();
            } else {
                gt_path = gt_argument;
            }
        }

        std::string mask_path;
        if (!mask_argument.empty()) {
            const fs::path mask_base(mask_argument);
            if (fs::is_directory(mask_base)) {
                const auto match = findMatchingFile(mask_base, stem, false);
                if (match.has_value()) mask_path = match->string();
            } else {
                mask_path = mask_argument;
            }
        }

        const ProcessResult result = processDepth(
            depth, stem, sample_output, camera, estimation,
            gt_path, gt_format, mask_path,
            ignore_sign, save_float, heatmap_max);

        if (!result.success) {
            std::cerr << "[失败] " << stem << ": " << result.message << '\n';
        } else {
            ++success_count;
            if (result.has_metrics) {
                ++evaluated_count;
                weighted_angle_sum += result.metrics.mean_deg * result.metrics.valid_count;
                total_valid_pixels += result.metrics.valid_count;
            } else if (!gt_argument.empty()) {
                std::cerr << "  [警告] 未找到同名 GT，已仅生成预测结果。\n";
            }
        }

        if (csv.is_open()) {
            csv << '"' << stem << "\"," << (result.success ? "ok" : "failed") << ',';
            if (result.has_metrics) {
                csv << result.metrics.valid_count << ','
                    << result.metrics.mean_deg << ','
                    << result.metrics.median_deg << ','
                    << result.metrics.rmse_deg << ','
                    << result.metrics.pct_11_25 << ','
                    << result.metrics.pct_22_5 << ','
                    << result.metrics.pct_30 << ',';
            } else {
                csv << "0,0,0,0,0,0,0,";
            }
            csv << '"' << result.message << "\"\n";
        }
    }

    std::cout << "\n============================================================\n";
    std::cout << "成功处理: " << success_count << " / " << depth_files.size() << " 张\n";
    if (evaluated_count > 0 && total_valid_pixels > 0) {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "完成定量评测: " << evaluated_count << " 张\n";
        std::cout << "按有效像素加权的整体平均角度误差: "
                  << weighted_angle_sum / static_cast<double>(total_valid_pixels) << " deg\n";
    }
    std::cout << "输出目录: " << output_root << '\n';
    std::cout << "============================================================\n";

    return success_count == depth_files.size() ? 0 : 1;
}
