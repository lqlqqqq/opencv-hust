// color_detector.h
#ifndef COLOR_DETECTOR_H
#define COLOR_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <memory>

// 检测结果结构体
struct ColorDetectionResult {
    int id;                 // 检测ID
    std::string color;      // 检测到的颜色
    double center_x;        // 中心点X坐标
    double center_y;        // 中心点Y坐标
    double width;          // 宽度
    double height;         // 高度
    double area;           // 面积
    double aspect_ratio;   // 宽高比
    double circularity;    // 圆形度
    double confidence;     // 置信度
    double angle;          // 旋转角度
    
    ColorDetectionResult() 
        : id(0), color(""), center_x(0.0), center_y(0.0), 
          width(0.0), height(0.0), area(0.0), aspect_ratio(0.0),
          circularity(0.0), confidence(0.0), angle(0.0) {}
    
    ColorDetectionResult(int id, const std::string& color, 
                        double center_x, double center_y,
                        double width, double height, double area,
                        double aspect_ratio, double circularity,
                        double confidence, double angle)
        : id(id), color(color), center_x(center_x), center_y(center_y),
          width(width), height(height), area(area), aspect_ratio(aspect_ratio),
          circularity(circularity), confidence(confidence), angle(angle) {}
};

// 统计信息结构体
struct DetectionStatistics {
    int total_detections;          // 总检测数
    double avg_processing_time_ms; // 平均处理时间
    int min_area;                  // 最小面积配置
    double confidence_threshold;   // 置信度阈值配置
    std::string target_color;      // 目标颜色配置
};

// 颜色范围结构体
struct ColorRange {
    cv::Scalar lower;
    cv::Scalar upper;
    
    ColorRange(const cv::Scalar& lower, const cv::Scalar& upper)
        : lower(lower), upper(upper) {}
};

// 颜色检测器类
class ColorDetector {
public:
    // 构造函数
    ColorDetector(const std::string& target_color = "red",
                  int min_area = 100,
                  double confidence_threshold = 0.3,
                  bool enable_morphology = true,
                  int morph_kernel_size = 5);
    
    // 从HSV图像检测
    std::pair<std::vector<ColorDetectionResult>, cv::Mat> 
    detect(const cv::Mat& hsv_image, bool verbose = true);
    
    // 从BGR图像检测
    std::pair<std::vector<ColorDetectionResult>, cv::Mat> 
    detect_from_bgr(const cv::Mat& bgr_image, bool verbose = true);
    
    // 获取统计信息
    DetectionStatistics get_statistics() const;
    
    // 重置统计信息
    void reset_statistics();
    
    // 设置参数
    void set_target_color(const std::string& color);
    void set_min_area(int min_area);
    void set_confidence_threshold(double threshold);
    void set_morphology_enabled(bool enabled);
    void set_morph_kernel_size(int size);
    
    // 获取参数
    std::string get_target_color() const { return target_color_; }
    int get_min_area() const { return min_area_; }
    double get_confidence_threshold() const { return confidence_threshold_; }
    bool is_morphology_enabled() const { return enable_morphology_; }
    int get_morph_kernel_size() const { return morph_kernel_size_; }
    
    // 快速检测函数（静态方法）
    static std::pair<std::vector<ColorDetectionResult>, cv::Mat> 
    detect_color_with_mask(const cv::Mat& image,
                          const std::string& target_color = "red",
                          int min_area = 100,
                          bool verbose = true);
    
private:
    // 私有方法
    cv::Mat create_color_mask(const cv::Mat& hsv_image);
    ColorDetectionResult analyze_contour(const std::vector<cv::Point>& contour, int detection_id);
    double calculate_circularity(const std::vector<cv::Point>& contour) const;
    void print_detection_info(const std::vector<ColorDetectionResult>& detections,
                             double processing_time_ms, 
                             const cv::Mat& mask,
                             bool verbose) const;
    
    // 私有数据成员
    std::string target_color_;
    int min_area_;
    double confidence_threshold_;
    bool enable_morphology_;
    int morph_kernel_size_;
    
    // 形态学核
    cv::Mat morph_kernel_;
    
    // 颜色范围映射
    static std::map<std::string, std::vector<std::pair<cv::Scalar, cv::Scalar>>> color_ranges_;
    
    // 统计信息
    int total_detections_;
    std::vector<double> processing_times_;
    
    // 初始化颜色范围
    static void initialize_color_ranges();
};

#endif // COLOR_DETECTOR_H