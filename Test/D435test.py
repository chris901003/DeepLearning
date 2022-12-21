import pyrealsense2 as rs
import numpy as np
import cv2
import time


def show_colorizer_depth_img():
    colorizer = rs.colorizer()
    hole_filling = rs.hole_filling_filter()
    filled_depth = hole_filling.process(depth_frame)
    colorized_depth = np.asanyarray(colorizer.colorize(filled_depth).get_data())
    cv2.imshow('filled depth', colorized_depth)


if __name__ == "__main__":
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # Start streaming
    pipeline.start(config)
    # 深度图像向彩色对齐
    align_to_color = rs.align(rs.stream.color)

    pTime = 0
    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()

            frames = align_to_color.process(frames)

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(color_image, f"FPS : {int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            cv2.imshow('rgb_img', color_image)

            show_colorizer_depth_img()

            # Press esc or 'q' to close the image window
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        # Stop streaming
        pipeline.stop()
