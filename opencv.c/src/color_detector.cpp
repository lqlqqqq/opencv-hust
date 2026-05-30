// color_detector.cpp
#include "color_detector.h"
#include <iostream>
#include <cmath>
#include <iomanip>

// 初始化静态成员变量
std::map<std::string, std::vector<std::pair<cv::Scalar, cv::Scalar>>> ColorDetector::color_ranges_;

// 构造函数
ColorDetector::ColorDetector(const std::string& target_color,
                             int min_area,
                             double confidence_threshold,
                             bool enable_morphology,
                             int morph_kernel_size)
    : target_color_(target_color),
      min_area_(min_area),
      confidence_threshold_(confidence_threshold),
      enable_morphology_(enable_morphology),
      morph_kernel_size_(morph_kernel_size),
      total_detections_(0) {
    
    // 初始化颜色范围
    initialize_color_ranges();
    
    // 检查目标颜色是否支持
    if (color_ranges_.find(target_color_) == color_ranges_.end()) {
        std::cerr << "警告: 不支持的颜色 '" << target_color_ 
                  << "', 默认使用红色" << std::endl;
        target_color_ = "red";
    }
    
    // 创建形态学核
    if (enable_morphology_) {
        morph_kernel_ = cv::getStructuringElement(
            cv::MORPH_ELLIPSE, 
            cv::Size(morph_kernel_size_, morph_kernel_size_)
        );
    }
    
    // 初始化处理时间记录
    processing_times_.reserve(100);  // 预分配空间
}

// 初始化颜色范围
void ColorDetector::initialize_color_ranges() {
    static bool initialized = false;
    if (initialized) return;
    
    // 红色（两个范围）
    color_ranges_["red"] = {
        {cv::Scalar(0, 100, 100), cv::Scalar(10, 255, 255)},
        {cv::Scalar(160, 100, 100), cv::Scalar(179, 255, 255)}
    };
    
    // 绿色
    color_ranges_["green"] = {
        {cv::Scalar(40, 100, 100), cv::Scalar(80, 255, 255)}
    };
    
    // 蓝色
    color_ranges_["blue"] = {
        {cv::Scalar(100, 100, 100), cv::Scalar(130, 255, 255)}
    };
    
    // 黄色
    color_ranges_["yellow"] = {
        {cv::Scalar(20, 100, 100), cv::Scalar(40, 255, 255)}
    };
    
    // 橙色
    color_ranges_["orange"] = {
        {cv::Scalar(10, 100, 100), cv::Scalar(20, 255, 255)}
    };
    
    // 紫色
    color_ranges_["purple"] = {
        {cv::Scalar(130, 100, 100), cv::Scalar(150, 255, 255)}
    };
    
    initialized = true;
}

// 从HSV图像检测
std::pair<std::vector<ColorDetectionResult>, cv::Mat> 
ColorDetector::detect(const cv::Mat& hsv_image, bool verbose) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::vector<ColorDetectionResult> detections;
    cv::Mat mask;
    
    // 检查输入图像
    if (hsv_image.empty()) {
        std::cerr << "错误: 输入图像为空" << std::endl;
        return {detections, mask};
    }
    
    if (hsv_image.channels() != 3) {
        std::cerr << "错误: 输入图像不是3通道HSV图像" << std::endl;
        return {detections, mask};
    }
    
    // 1. 创建颜色掩码
    mask = create_color_mask(hsv_image);
    if (mask.empty()) {
        std::cerr << "错误: 创建颜色掩码失败" << std::endl;
        return {detections, mask};
    }
    
    // 2. 形态学优化
    if (enable_morphology_ && !morph_kernel_.empty()) {
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, morph_kernel_);
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, morph_kernel_);
    }
    
    // 3. 查找轮廓
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(mask, contours, hierarchy, 
                     cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    // 4. 分析轮廓
    int detection_id = 1;
    for (const auto& contour : contours) {
        ColorDetectionResult detection = analyze_contour(contour, detection_id);
        if (detection.area > 0 && detection.confidence >= confidence_threshold_) {
            detections.push_back(detection);
            detection_id++;
        }
    }
    
    // 5. 计算处理时间
    auto end_time = std::chrono::high_resolution_clock::now();
    double processing_time_ms = std::chrono::duration<double, std::milli>(
        end_time - start_time).count();
    
    // 6. 更新统计信息
    processing_times_.push_back(processing_time_ms);
    total_detections_ += detections.size();
    
    // 7. 输出信息
    if (verbose) {
        print_detection_info(detections, processing_time_ms, mask, true);
    }
    
    return {detections, mask};
}

// 从BGR图像检测
std::pair<std::vector<ColorDetectionResult>, cv::Mat> 
ColorDetector::detect_from_bgr(const cv::Mat& bgr_image, bool verbose) {
    if (bgr_image.empty()) {
        std::cerr << "错误: 输入图像为空" << std::endl;
        return {{}, cv::Mat()};
    }
    
    // 转换为HSV
    cv::Mat hsv_image;
    cv::cvtColor(bgr_image, hsv_image, cv::COLOR_BGR2HSV);
    
    return detect(hsv_image, verbose);
}

// 创建颜色掩码
cv::Mat ColorDetector::create_color_mask(const cv::Mat& hsv_image) {
    auto it = color_ranges_.find(target_color_);
    if (it == color_ranges_.end()) {
        std::cerr << "错误: 未找到颜色范围: " << target_color_ << std::endl;
        return cv::Mat();
    }
    
    cv::Mat mask = cv::Mat::zeros(hsv_image.size(), CV_8UC1);
    const auto& ranges = it->second;
    
    for (const auto& range : ranges) {
        cv::Mat temp_mask;
        cv::inRange(hsv_image, range.first, range.second, temp_mask);
        cv::bitwise_or(mask, temp_mask, mask);
    }
    
    return mask;
}

// 分析轮廓
ColorDetectionResult ColorDetector::analyze_contour(
    const std::vector<cv::Point>& contour, int detection_id) {
    
    // 计算面积
    double area = cv::contourArea(contour);
    if (area < min_area_) {
        return ColorDetectionResult();  // 返回空结果
    }
    
    // 计算最小外接矩形
    cv::RotatedRect rotated_rect = cv::minAreaRect(contour);
    cv::Point2f center = rotated_rect.center;
    cv::Size2f size = rotated_rect.size;
    float angle = rotated_rect.angle;
    
    // 确保宽度和高度正确
    float width = size.width;
    float height = size.height;
    if (width < height) {
        std::swap(width, height);
        angle += 90.0f;  // 角度调整
    }
    
    // 计算宽高比
    double aspect_ratio = (height > 0) ? (width / height) : 0.0;
    
    // 计算圆形度
    double circularity = calculate_circularity(contour);
    
    // 计算置信度
    double confidence = std::min(circularity, 1.0);
    
    return ColorDetectionResult(
        detection_id,
        target_color_,
        center.x, center.y,
        width, height,
        area,
        aspect_ratio,
        circularity,
        confidence,
        static_cast<double>(angle)
    );
}

// 计算圆形度
double ColorDetector::calculate_circularity(const std::vector<cv::Point>& contour) const {
    double area = cv::contourArea(contour);
    double perimeter = cv::arcLength(contour, true);
    
    if (perimeter <= 0) {
        return 0.0;
    }
    
    // 圆形度 = 4π * 面积 / 周长²
    double circularity = 4.0 * CV_PI * area / (perimeter * perimeter);
    
    // 限制在0-1之间
    return std::min(std::max(circularity, 0.0), 1.0);
}

// 打印检测信息
void ColorDetector::print_detection_info(
    const std::vector<ColorDetectionResult>& detections,
    double processing_time_ms,
    const cv::Mat& mask,
    bool verbose) const {
    
    if (!verbose) return;
    
    std::cout << std::fixed << std::setprecision(1);
    
    if (detections.empty()) {
        std::cout << "未检测到 " << target_color_ << " 目标" << std::endl;
    } else {
        std::cout << "检测到 " << detections.size() << " 个 " 
                  << target_color_ << " 目标" << std::endl;
    }
    
    std::cout << "处理时间: " << processing_time_ms << "ms" << std::endl;
    std::cout << "掩码尺寸: " << mask.cols << "x" << mask.rows;
    
    if (!mask.empty()) {
        int non_zero = cv::countNonZero(mask);
        std::cout << ", 非零像素: " << non_zero;
    }
    std::cout << std::endl;
    
    if (!detections.empty()) {
        std::cout << std::string(40, '-') << std::endl;
        std::cout << std::fixed << std::setprecision(0);
        
        for (const auto& det : detections) {
            std::cout << "  ID:" << det.id 
                      << " 中心(" << det.center_x << "," << det.center_y << ")"
                      << " 尺寸:" << det.width << "x" << det.height
                      << " 面积:" << det.area
                      << " 置信度:" << std::setprecision(2) << det.confidence
                      << std::setprecision(0) << std::endl;
        }
    }
}

// 获取统计信息
DetectionStatistics ColorDetector::get_statistics() const {
    DetectionStatistics stats;
    stats.total_detections = total_detections_;
    stats.min_area = min_area_;
    stats.confidence_threshold = confidence_threshold_;
    stats.target_color = target_color_;
    
    if (!processing_times_.empty()) {
        double sum = 0.0;
        for (double time : processing_times_) {
            sum += time;
        }
        stats.avg_processing_time_ms = sum / processing_times_.size();
    } else {
        stats.avg_processing_time_ms = 0.0;
    }
    
    return stats;
}

// 重置统计信息
void ColorDetector::reset_statistics() {
    total_detections_ = 0;
    processing_times_.clear();
}

// 设置参数
void ColorDetector::set_target_color(const std::string& color) {
    target_color_ = color;
    if (color_ranges_.find(target_color_) == color_ranges_.end()) {
        std::cerr << "警告: 不支持的颜色 '" << color 
                  << "', 使用红色" << std::endl;
        target_color_ = "red";
    }
}

void ColorDetector::set_min_area(int min_area) {
    min_area_ = (min_area > 0) ? min_area : 1;
}

void ColorDetector::set_confidence_threshold(double threshold) {
    confidence_threshold_ = std::max(0.0, std::min(1.0, threshold));
}

void ColorDetector::set_morphology_enabled(bool enabled) {
    enable_morphology_ = enabled;
    if (enabled && morph_kernel_.empty()) {
        morph_kernel_ = cv::getStructuringElement(
            cv::MORPH_ELLIPSE, 
            cv::Size(morph_kernel_size_, morph_kernel_size_)
        );
    }
}

void ColorDetector::set_morph_kernel_size(int size) {
    morph_kernel_size_ = (size > 0) ? size : 1;
    if (enable_morphology_) {
        morph_kernel_ = cv::getStructuringElement(
            cv::MORPH_ELLIPSE, 
            cv::Size(morph_kernel_size_, morph_kernel_size_)
        );
    }
}

// 快速检测函数
std::pair<std::vector<ColorDetectionResult>, cv::Mat> 
ColorDetector::detect_color_with_mask(const cv::Mat& image,
                                     const std::string& target_color,
                                     int min_area,
                                     bool verbose) {
    
    // 创建临时检测器
    ColorDetector detector(target_color, min_area);
    
    // 判断图像类型
    if (image.channels() == 3) {
        // 假设是BGR图像
        return detector.detect_from_bgr(image, verbose);
    } else if (image.channels() == 1) {
        // 灰度图，需要先转换为BGR
        cv::Mat bgr_image;
        cv::cvtColor(image, bgr_image, cv::COLOR_GRAY2BGR);
        return detector.detect_from_bgr(bgr_image, verbose);
    } else {
        std::cerr << "错误: 不支持的图像通道数: " << image.channels() << std::endl;
        return {{}, cv::Mat()};
    }
}