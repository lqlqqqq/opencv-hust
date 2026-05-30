
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <chrono>
#include <memory>
#include <opencv2/opencv.hpp>

// 包含自定义头文件（暂定，后续跟着实际代码改）
#include "color_detector.h"
#include "preprocessor.h"
#include "mask_utils.h"
#include "file_utils.h"

namespace fs = std::filesystem;

// 配置结构体
struct DetectionConfig {
    // 检测开关
    bool enable_color_detection = true;
    bool enable_shape_detection = true;
    bool detect_rectangles = true;
    bool detect_circles = true;
    bool use_preprocess = false;
    bool show_process = false;
    bool mask_applied_image = true;
    
    // 颜色检测参数
    std::string target_color = "red";
    int min_area = 50;
    double confidence_threshold = 0.3;
    
    // 形状检测参数
    int min_rectangle_area = 100;
    int min_circle_area = 200;
    double min_circularity = 0.8;
    double max_aspect_ratio_diff = 0.2;
    
    // 文件路径
    std::string input_image = "data/test002.png";
    std::string output_dir = "result";
};

// 结果结构体
struct DetectionResults {
    std::vector<ColorDetectionResult> color_detections;
    cv::Mat color_mask;
    
    // 形状检测结果（暂时占位，后面实现）
    // ShapeDetectionResult rectangle_result;
    // ShapeDetectionResult circle_result;
    
    bool success = false;
    std::string error_message;
};

// 确保结果文件夹存在
bool ensure_result_folder(const std::string& result_dir) {
    try {
        if (!fs::exists(result_dir)) {
            bool created = fs::create_directories(result_dir);
            if (created) {
                std::cout << "✓ 创建结果文件夹: " << result_dir << std::endl;
                return true;
            } else {
                std::cerr << "✗ 无法创建结果文件夹: " << result_dir << std::endl;
                return false;
            }
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "✗ 创建结果文件夹时出错: " << e.what() << std::endl;
        return false;
    }
}

// 获取结果文件夹下的完整文件路径
std::string get_result_path(const std::string& result_dir, const std::string& filename) {
    return (fs::path(result_dir) / filename).string();
}

// 主程序
DetectionResults main_program(const DetectionConfig& config) {
    DetectionResults results;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        std::cout << "=== C++ OpenCV 视觉检测系统 ===" << std::endl;
        std::cout << "开始时间: " << std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()) << std::endl;
        
        // 1. 确保结果文件夹存在
        std::string result_dir = config.output_dir;
        if (!ensure_result_folder(result_dir)) {
            results.error_message = "无法创建结果文件夹";
            return results;
        }
        
        // 2. 读取图片
        std::cout << "\n1. 读取图片..." << std::endl;
        std::string current_dir = fs::current_path().string();
        std::string image_path = config.input_image;
        
        // 如果是相对路径，转换为绝对路径
        if (!fs::path(image_path).is_absolute()) {
            image_path = (fs::path(current_dir) / image_path).string();
        }
        
        std::cout << "   图片路径: " << image_path << std::endl;
        
        cv::Mat original_image = cv::imread(image_path);
        if (original_image.empty()) {
            std::cerr << "错误: 无法读取图片 " << image_path << std::endl;
            results.error_message = "无法读取图片";
            return results;
        }
        
        std::cout << "✓ 图片尺寸: " << original_image.cols << "x" << original_image.rows << std::endl;
        
        // 保存原始图片
        cv::imwrite(get_result_path(result_dir, "original.jpg"), original_image);
        
        // 3. 预处理
        cv::Mat working_image = original_image.clone();
        
        if (config.use_preprocess) {
            std::cout << "\n2. 执行图像预处理..." << std::endl;
            
            try {
                // 创建预处理器
                Preprocessor preprocessor;
                std::cout << "  a) 图像预处理（降噪、对比度增强、锐化）..." << std::endl;
                cv::Mat preprocessed_image = preprocessor.process(working_image);
                
                std::cout << "✓ 预处理完成" << std::endl;
                
                // 保存预处理结果
                cv::imwrite(get_result_path(result_dir, "preprocessed.jpg"), preprocessed_image);
                
                // 显示结果
                if (config.show_process) {
                    cv::imshow("原始图像", original_image);
                    cv::imshow("预处理后图像", preprocessed_image);
                    cv::waitKey(100);
                }
                
                working_image = preprocessed_image;
                
            } catch (const std::exception& e) {
                std::cerr << "✗ 预处理失败: " << e.what() << std::endl;
                std::cout << "  将使用原始图像" << std::endl;
            }
        } else {
            std::cout << "\n2. 跳过预处理，使用原始图像" << std::endl;
        }
        
        // 4. 颜色检测
        if (config.enable_color_detection) {
            std::cout << "\n3. 执行颜色检测..." << std::endl;
            
            ColorDetector detector(
                config.target_color,
                config.min_area,
                config.confidence_threshold
            );
            
            auto [detections, mask] = detector.detect_from_bgr(working_image, true);
            results.color_detections = detections;
            results.color_mask = mask;
            
            if (!detections.empty()) {
                std::cout << "✓ 颜色检测: 总共检测到 " << detections.size() << " 个目标" << std::endl;
            } else {
                std::cout << "✗ 颜色检测: 未检测到目标" << std::endl;
            }
            
            // 保存颜色掩码
            if (!mask.empty()) {
                cv::imwrite(get_result_path(result_dir, "color_mask.jpg"), mask);
            }
            
            // 绘制检测结果
            cv::Mat color_detection_image = working_image.clone();
            for (const auto& det : detections) {
                int x = static_cast<int>(det.center_x - det.width / 2);
                int y = static_cast<int>(det.center_y - det.height / 2);
                int w = static_cast<int>(det.width);
                int h = static_cast<int>(det.height);
                
                // 绘制边界框
                cv::rectangle(color_detection_image, cv::Rect(x, y, w, h), cv::Scalar(0, 255, 0), 2);
                
                // 绘制中心点
                cv::circle(color_detection_image, cv::Point(det.center_x, det.center_y), 3, cv::Scalar(0, 0, 255), -1);
                
                // 绘制标签
                std::string label = config.target_color + ":" + std::to_string(det.id);
                cv::putText(color_detection_image, label, cv::Point(x, y - 5), 
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
            }
            
            cv::imwrite(get_result_path(result_dir, "color_detection_result.jpg"), color_detection_image);
            
            if (config.show_process && !mask.empty()) {
                cv::imshow("颜色掩码", mask);
                cv::waitKey(100);
            }
        } else {
            std::cout << "\n3. 跳过颜色检测" << std::endl;
        }
        
        // 5. 形状检测（暂时跳过，后面实现）
        if (config.enable_shape_detection) {
            std::cout << "\n4. 执行图形检测..." << std::endl;
            std::cout << std::string(50, '-') << std::endl;
            
            // TODO: 实现矩形检测
            if (config.detect_rectangles) {
                std::cout << "检测矩形..." << std::endl;
                // rectangle_result = detect_rectangles(...)
                std::cout << "  ✗ 矩形检测暂未实现" << std::endl;
            }
            
            // TODO: 实现圆形检测
            if (config.detect_circles) {
                std::cout << "\n检测圆形..." << std::endl;
                // circle_result = detect_circles(...)
                std::cout << "  ✗ 圆形检测暂未实现" << std::endl;
            }
            
            std::cout << std::string(50, '-') << std::endl;
        }
        
        // 6. 应用掩码到原图
        if (config.mask_applied_image && !results.color_mask.empty()) {
            std::cout << "\n5. 应用掩码到原始图像..." << std::endl;
            
            try {
                // 使用MaskUtils应用掩码
                MaskResults mask_results = MaskUtils::apply_masks(original_image, results.color_mask);
                
                if (!mask_results.mask_applied_image.empty()) {
                    std::cout << "✓ 掩码应用成功" << std::endl;
                    
                    // 保存结果
                    cv::imwrite(get_result_path(result_dir, "mask_applied_result.jpg"), 
                               mask_results.mask_applied_image);
                    
                    if (!mask_results.combined_mask.empty()) {
                        cv::imwrite(get_result_path(result_dir, "combined_mask.jpg"), 
                                   mask_results.combined_mask);
                        std::cout << "  ✓ 合并掩码已保存" << std::endl;
                    }
                    
                    if (!mask_results.color_only_image.empty()) {
                        cv::imwrite(get_result_path(result_dir, "color_only_result.jpg"), 
                                   mask_results.color_only_image);
                    }
                    
                    // 显示结果
                    if (config.show_process) {
                        cv::imshow("掩码处理结果", mask_results.mask_applied_image);
                        cv::imshow("仅颜色结果", mask_results.color_only_image);
                        cv::waitKey(100);
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << "✗ 掩码应用失败: " << e.what() << std::endl;
            }
        }
        
        // 7. 计算总耗时
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        // 8. 列出所有保存的文件
        std::cout << "\n所有检测结果已保存到: " << result_dir << std::endl;
        std::cout << "保存的文件:" << std::endl;
        
        try {
            for (const auto& entry : fs::directory_iterator(result_dir)) {
                if (entry.is_regular_file()) {
                    auto file_size = entry.file_size();
                    std::cout << "  - " << entry.path().filename().string() 
                              << " (" << file_size << " 字节)" << std::endl;
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "  无法列出文件: " << e.what() << std::endl;
        }
        
        std::cout << "\n处理完成，总耗时: " << duration.count() << "ms" << std::endl;
        std::cout << "\n检测完成，按任意键关闭所有窗口..." << std::endl;
        
        results.success = true;
        
    } catch (const std::exception& e) {
        std::cerr << "\n程序执行出错: " << e.what() << std::endl;
        results.error_message = e.what();
    }
    
    return results;
}

// 主函数
int main(int argc, char** argv) {
    // 配置参数
    DetectionConfig config;
    
    // 如果命令行有参数，使用第一个参数作为图片路径
    if (argc > 1) {
        config.input_image = argv[1];
    }
    
    // 可以在这里修改配置
    config.enable_color_detection = true;
    config.enable_shape_detection = false;  // 暂时关闭形状检测
    config.use_preprocess = false;
    config.show_process = true;
    config.mask_applied_image = true;
    
    // 运行主程序
    DetectionResults results = main_program(config);
    
    if (results.success) {
        if (!results.color_detections.empty()) {
            std::cout << "\n=== 检测结果汇总 ===" << std::endl;
            std::cout << "颜色检测: 找到 " << results.color_detections.size() << " 个目标" << std::endl;
            
            for (const auto& det : results.color_detections) {
                std::cout << "  目标" << det.id << ": "
                          << "中心(" << det.center_x << ", " << det.center_y << "), "
                          << "尺寸" << det.width << "x" << det.height << ", "
                          << "面积" << det.area << ", "
                          << "置信度" << det.confidence << std::endl;
            }
        }
        
        std::cout << "\n程序执行完成" << std::endl;
        
        // 等待按键关闭窗口
        if (config.show_process) {
            cv::waitKey(0);
        }
        
        cv::destroyAllWindows();
        return 0;
    } else {
        std::cerr << "程序执行失败: " << results.error_message << std::endl;
        return 1;
    }
}