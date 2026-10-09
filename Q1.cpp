#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

#include <opencv2/opencv.hpp>

#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>

class DetectorNode : public rclcpp::Node
{
public:
    DetectorNode() : Node("detector_node")
    {
        pub_ = this->create_publisher<std_msgs::msg::String>(
            "/detection", rclcpp::QoS(10).reliable());

        cap_.open(0);
        if (!cap_.isOpened()) {
            RCLCPP_ERROR(this->get_logger(), "cannot open camera 0");
        } else {
            cap_.set(cv::CAP_PROP_FRAME_WIDTH, 640);
            cap_.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
            cap_.set(cv::CAP_PROP_BUFFERSIZE, 1);
            RCLCPP_INFO(this->get_logger(), "camera opened");
        }

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(20),
            std::bind(&DetectorNode::tick, this));

        RCLCPP_INFO(this->get_logger(), "detector_node started");
    }

    ~DetectorNode()
    {
        if (cap_.isOpened())
            cap_.release();
    }

private:
    struct RectInfo {
        cv::Rect rect;
        std::string color;   // "RED" or "BLUE"
    };

    void tick()
    {
        if (!cap_.isOpened())
            return;

        cv::Mat frame;
        cap_ >> frame;
        if (frame.empty())
            return;

        // ---- 1) 颜色分割 ----
        cv::Mat hsv;
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

        cv::Mat mask_red1, mask_red2, mask_red;
        // 红色阈值只做很小放宽：H 上限 10→12，S/V 下限 100→80
        cv::inRange(hsv, cv::Scalar(0,   80, 80),
                         cv::Scalar(12,  255, 255), mask_red1);
        cv::inRange(hsv, cv::Scalar(160, 80, 80),
                         cv::Scalar(180, 255, 255), mask_red2);
        cv::bitwise_or(mask_red1, mask_red2, mask_red);

        cv::Mat mask_blue;
        cv::inRange(hsv, cv::Scalar(100, 80, 80),
                         cv::Scalar(130, 255, 255), mask_blue);

        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::morphologyEx(mask_red,  mask_red,  cv::MORPH_OPEN,  kernel);
        cv::morphologyEx(mask_red,  mask_red,  cv::MORPH_CLOSE, kernel);
        cv::morphologyEx(mask_blue, mask_blue, cv::MORPH_OPEN,  kernel);
        cv::morphologyEx(mask_blue, mask_blue, cv::MORPH_CLOSE, kernel);

        // ---- 2) 分别提取红块和蓝块 ----
        std::vector<RectInfo> reds  = extract_rects(mask_red,  "RED");
        std::vector<RectInfo> blues = extract_rects(mask_blue, "BLUE");

        // ---- 3) 配对：找红蓝上下拼接的正方形 ----
        std::vector<RectInfo> merged;          // 拼接后的整体
        std::vector<bool> red_used(reds.size(), false);
        std::vector<bool> blue_used(blues.size(), false);

        for (size_t i = 0; i < reds.size(); ++i) {
            for (size_t j = 0; j < blues.size(); ++j) {
                if (red_used[i] || blue_used[j])
                    continue;
                if (is_adjacent_halves(reds[i].rect, blues[j].rect)) {
                    // 合并成一个整体矩形
                    cv::Rect whole = reds[i].rect | blues[j].rect;
                    merged.push_back({whole, "RED_BLUE"});
                    red_used[i]  = true;
                    blue_used[j] = true;
                    break;
                }
            }
        }

        // ---- 4) 组消息 ----
        std::string result = "detected:";

        // 先报拼接的
        for (const auto& m : merged) {
            draw_rect(frame, m.rect, "RED_BLUE", cv::Scalar(0, 255, 0));
            char buf[128];
            snprintf(buf, sizeof(buf), " RED_BLUE(%d,%d,%d,%d)",
                     m.rect.x, m.rect.y, m.rect.width, m.rect.height);
            result += buf;
        }

        // 再报单独的红块
        for (size_t i = 0; i < reds.size(); ++i) {
            if (red_used[i]) continue;
            draw_rect(frame, reds[i].rect, "RED", cv::Scalar(0, 0, 255));
            char buf[128];
            snprintf(buf, sizeof(buf), " RED(%d,%d,%d,%d)",
                     reds[i].rect.x, reds[i].rect.y,
                     reds[i].rect.width, reds[i].rect.height);
            result += buf;
        }

        // 再报单独的蓝块
        for (size_t j = 0; j < blues.size(); ++j) {
            if (blue_used[j]) continue;
            draw_rect(frame, blues[j].rect, "BLUE", cv::Scalar(255, 0, 0));
            char buf[128];
            snprintf(buf, sizeof(buf), " BLUE(%d,%d,%d,%d)",
                     blues[j].rect.x, blues[j].rect.y,
                     blues[j].rect.width, blues[j].rect.height);
            result += buf;
        }

        if (result == "detected:")
            result += " nothing";

        std_msgs::msg::String msg;
        msg.data = result;
        pub_->publish(msg);
        RCLCPP_INFO(this->get_logger(), "%s", msg.data.c_str());

        // ---- 5) 调试预览 ----
        cv::imshow("frame", frame);
        cv::imshow("mask_red",  mask_red);
        cv::imshow("mask_blue", mask_blue);
        cv::waitKey(1);
    }

    // 从掩膜里提取所有近似矩形的轮廓
    std::vector<RectInfo> extract_rects(const cv::Mat& mask,
                                        const std::string& color)
    {
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL,
                         cv::CHAIN_APPROX_SIMPLE);

        std::vector<RectInfo> out;
        for (const auto& c : contours) {
            double area = cv::contourArea(c);
            if (area < 1500)
                continue;

            cv::Rect r = cv::boundingRect(c);
            float ratio = (float)r.width / (float)r.height;
            // 放宽：允许横长条（上下拼接时单块是横长条）
            if (ratio < 0.3f || ratio > 3.0f)
                continue;

            out.push_back({r, color});
        }
        return out;
    }

    // 判断红块和蓝块是否上下拼成一个 1:1 的正方形
    // 条件：上下紧贴、宽度接近、高度各占一半左右、整体接近 1:1
    bool is_adjacent_halves(const cv::Rect& red, const cv::Rect& blue)
    {
        // 宽度要接近（允许 15% 差异）
        float w_ratio = (float)std::min(red.width, blue.width) /
                        (float)std::max(red.width, blue.width);
        if (w_ratio < 0.85f)
            return false;

        // 水平方向要对齐（中心 x 接近）
        int red_cx  = red.x + red.width / 2;
        int blue_cx = blue.x + blue.width / 2;
        if (std::abs(red_cx - blue_cx) > std::max(red.width, blue.width) * 0.3)
            return false;

        // 垂直方向必须相邻：红在上蓝在下，或蓝在上红在下
        bool red_top  = (red.y + red.height >= blue.y - 10) &&
                        (red.y + red.height <= blue.y + 20);
        bool blue_top = (blue.y + blue.height >= red.y - 10) &&
                        (blue.y + blue.height <= red.y + 20);
        if (!red_top && !blue_top)
            return false;

        // 高度各占一半左右（允许 30% 偏差）
        float h_ratio = (float)std::min(red.height, blue.height) /
                        (float)std::max(red.height, blue.height);
        if (h_ratio < 0.7f)
            return false;

        // 整体宽高比接近 1:1
        cv::Rect whole = red | blue;
        float whole_ratio = (float)whole.width / (float)whole.height;
        if (whole_ratio < 0.85f || whole_ratio > 1.15f)
            return false;

        return true;
    }

    void draw_rect(cv::Mat& frame, const cv::Rect& r,
                   const std::string& label, const cv::Scalar& color)
    {
        cv::rectangle(frame, r, color, 2);
        cv::putText(frame, label,
                    cv::Point(r.x, r.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
    }

    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    cv::VideoCapture cap_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DetectorNode>());
    rclcpp::shutdown();
    return 0;
}