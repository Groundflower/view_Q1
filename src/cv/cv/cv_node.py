import math

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


class DetectorNode(Node):
    def __init__(self):
        super().__init__("detector_node")
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.RELIABLE
        self.pub_ = self.create_publisher(String, "/detection", qos)
        self.cap_ = cv2.VideoCapture(0)
        if not self.cap_.isOpened():
            self.get_logger().error("cannot open camera 0")
        else:
            self.cap_.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap_.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap_.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.get_logger().info("camera opened")
        self.timer_ = self.create_timer(0.02, self.tick)
        self.get_logger().info("detector_node started")
    def destroy_node(self):
        if self.cap_.isOpened():
            self.cap_.release()
        cv2.destroyAllWindows()
        super().destroy_node()
    def tick(self):
        if not self.cap_.isOpened():
            return
        ok, frame = self.cap_.read()
        if not ok or frame is None:
            return
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_red1 = cv2.inRange(hsv, (0, 80, 80), (12, 255, 255))
        mask_red2 = cv2.inRange(hsv, (160, 80, 80), (180, 255, 255))
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        mask_blue = cv2.inRange(hsv, (100, 80, 80), (130, 255, 255))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)
        # 提取
        reds = self.extract_rects(mask_red, "RED")
        blues = self.extract_rects(mask_blue, "BLUE")

        # 配对
        merged = []
        red_used = [False] * len(reds)
        blue_used = [False] * len(blues)

        for i, r in enumerate(reds):
            for j, b in enumerate(blues):
                if red_used[i] or blue_used[j]:
                    continue
                if self.is_adjacent_halves(r["rect"], b["rect"]):
                    x1 = min(r["rect"][0], b["rect"][0])
                    y1 = min(r["rect"][1], b["rect"][1])
                    x2 = max(r["rect"][0] + r["rect"][2],
                             b["rect"][0] + b["rect"][2])
                    y2 = max(r["rect"][1] + r["rect"][3],
                             b["rect"][1] + b["rect"][3])
                    merged.append({"rect": (x1, y1, x2 - x1, y2 - y1),
                                   "color": "RED_BLUE"})
                    red_used[i] = True
                    blue_used[j] = True
                    break

        # 消息
        result = "detected:"

        for m in merged:
            self.draw_rect(frame, m["rect"], "RED_BLUE", (0, 255, 0))
            x, y, w, h = m["rect"]
            result += f" RED_BLUE({x},{y},{w},{h})"

        for i, r in enumerate(reds):
            if red_used[i]:
                continue
            self.draw_rect(frame, r["rect"], "RED", (0, 0, 255))
            x, y, w, h = r["rect"]
            result += f" RED({x},{y},{w},{h})"

        for j, b in enumerate(blues):
            if blue_used[j]:
                continue
            self.draw_rect(frame, b["rect"], "BLUE", (255, 0, 0))
            x, y, w, h = b["rect"]
            result += f" BLUE({x},{y},{w},{h})"

        if result == "detected:":
            result += " nothing"

        msg = String()
        msg.data = result
        self.pub_.publish(msg)
        self.get_logger().info(result)

        cv2.imshow("frame", frame)
        cv2.imshow("mask_red", mask_red)
        cv2.imshow("mask_blue", mask_blue)
        cv2.waitKey(1)

    def extract_rects(self, mask, color):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 1500:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if h == 0:
                continue
            ratio = w / h
            if ratio < 0.3 or ratio > 3.0:
                continue
            out.append({"rect": (x, y, w, h), "color": color})
        return out

    def is_adjacent_halves(self, red, blue):
        rx, ry, rw, rh = red
        bx, by, bw, bh = blue

        # 宽度接近（ 15% 差异）
        w_ratio = min(rw, bw) / max(rw, bw)
        if w_ratio < 0.85:
            return False

        # 水平对齐（ x 接近）
        rcx = rx + rw // 2
        bcx = bx + bw // 2
        if abs(rcx - bcx) > max(rw, bw) * 0.3:
            return False
        # 仅分上下拼接
        gap_red_top = by - (ry + rh)      # 红在上
        gap_blue_top = ry - (by + bh)     # 蓝在上
        if not (-10 <= gap_red_top <= 20 or -10 <= gap_blue_top <= 20):
            return False

        # 高度（允许 30% 偏差）
        h_ratio = min(rh, bh) / max(rh, bh)
        if h_ratio < 0.7:
            return False

        # 整体宽高比接近 1:1
        x1 = min(rx, bx)
        y1 = min(ry, by)
        x2 = max(rx + rw, bx + bw)
        y2 = max(ry + rh, by + bh)
        whole_w = x2 - x1
        whole_h = y2 - y1
        if whole_h == 0:
            return False
        whole_ratio = whole_w / whole_h
        if whole_ratio < 0.85 or whole_ratio > 1.15:
            return False

        return True

    def draw_rect(self, frame, rect, label, color):
        x, y, w, h = rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()