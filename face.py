import os
import sys
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from tkinter import ttk
from PIL import Image
import io
import threading
import queue
import traceback
from pathlib import Path
from skimage import exposure

# 脸部宽度和鼻子位置的比例
FACE_WIDTH_RATIO = 0.52  # 脸部宽度占图片宽度的52%

# 有效的图片文件扩展名
VALID_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

def resource_path(relative_path):
    """
    获取在编译后的可执行文件中的资源文件的绝对路径。
    适用于 PyInstaller 打包后的运行环境。
    """
    try:
        # PyInstaller 创建的临时文件夹存放打包后的资源
        base_path = sys._MEIPASS
    except AttributeError:
        # 在常规环境中运行，基于脚本的位置
        base_path = Path(os.path.abspath("."))
    
    return (base_path / relative_path).resolve()

def rotate_image_if_needed(image):
    """
    如果图像是横向的（宽度 > 高度），则逆时针旋转90度。
    """
    height, width = image.shape[:2]
    if width > height:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image

def is_overexposed(image, threshold=240, proportion=0.1):
    """
    检测图像是否过曝。

    :param image: 输入图像 (BGR)
    :param threshold: 亮度阈值
    :param proportion: 亮度超过阈值的pixel比例阈值
    :return: 是否过曝 (True/False)
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv)
    overexposed_pixels = np.sum(v > threshold)
    total_pixels = v.size
    ratio = overexposed_pixels / total_pixels
    return ratio > proportion

def fix_overexposure(image, v_adjust):
    """
    修复过曝图像，通过降低过曝区域的亮度。

    :param image: 输入图像 (BGR)
    :param v_adjust: V 通道调整系数
    :return: 修复后的图像
    """
    # 将图像转换为 HSV 色彩空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_channel, s_channel, v_channel = cv2.split(hsv_image)

    # 对 V 通道进行亮度调整
    v_channel = np.clip(v_channel * v_adjust, 0, 255).astype(np.uint8)

    # 合并通道
    hsv_image = cv2.merge((h_channel, s_channel, v_channel))

    # 将图像转换回 BGR 色彩空间
    fixed_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return fixed_image

def get_face_landmarks(image, face_mesh):
    """
    使用 Mediapipe Face Mesh 获取人脸关键点。
    """
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None
    return results.multi_face_landmarks[0]

def align_face(image, landmarks):
    """
    对齐人脸，使其垂直。
    :param image: 输入图像 (BGR)
    :param landmarks: 人脸关键点
    :return: 对齐后图像
    """
    left_eye = landmarks.landmark[33]
    right_eye = landmarks.landmark[263]

    left_eye_center = (int(left_eye.x * image.shape[1]), int(left_eye.y * image.shape[0]))
    right_eye_center = (int(right_eye.x * image.shape[1]), int(right_eye.y * image.shape[0]))

    # 计算两个眼睛的中心点
    eye_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                  (left_eye_center[1] + right_eye_center[1]) // 2)

    # 计算眼睛连线与水平线的角度
    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dy, dx))

    # 获取旋转矩阵
    M = cv2.getRotationMatrix2D(eye_center, angle, scale=1)

    # 进行仿射变换，校正图像
    aligned_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

    return aligned_image

def get_nose_position(landmarks, image_width, image_height):
    """
    获取鼻尖的位置。
    """
    nose_tip = landmarks.landmark[1]
    x = int(nose_tip.x * image_width)
    y = int(nose_tip.y * image_height)
    return (x, y)

def get_face_width(landmarks, image_width, image_height):
    """
    通过左右脸颊的关键点计算脸部宽度。
    """
    left_cheek = landmarks.landmark[234]
    right_cheek = landmarks.landmark[454]
    x_left = int(left_cheek.x * image_width)
    x_right = int(right_cheek.x * image_width)
    face_width = abs(x_right - x_left)
    return face_width

def scale_image(image, face_width, target_face_width):
    """
    根据脸部宽度缩放图像，使脸部宽度达到目标值。
    """
    if face_width == 0:
        return image, 1.0  # 避免除以0
    scale_factor = target_face_width / face_width
    new_width = int(image.shape[1] * scale_factor)
    new_height = int(image.shape[0] * scale_factor)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return resized_image, scale_factor

def translate_image(image, current_nose_pos, desired_nose_pos, output_width, output_height):
    """
    平移图像，使鼻子位于指定的位置。
    """
    offset_x = desired_nose_pos[0] - current_nose_pos[0]
    offset_y = desired_nose_pos[1] - current_nose_pos[1]

    # 创建平移矩阵
    M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])

    # 应用平移
    translated_image = cv2.warpAffine(image, M, (output_width, output_height), borderMode=cv2.BORDER_REPLICATE)
    return translated_image

def crop_and_pad(image, output_width, output_height):
    """
    将图像裁剪或填充到指定尺寸，确保脸部不被裁剪。
    """
    height, width = image.shape[:2]

    # 如果图像小于目标尺寸，填充
    if width < output_width or height < output_height:
        delta_w = max(output_width - width, 0)
        delta_h = max(output_height - height, 0)
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REPLICATE)

    # 如果图像大于目标尺寸，裁剪
    height, width = image.shape[:2]
    if width > output_width or height > output_height:
        x_start = (width - output_width) // 2
        y_start = (height - output_height) // 2
        image = image[y_start:y_start + output_height, x_start:x_start + output_width]

    return image

def save_image_with_constraints(cv2_image, output_path, target_dpi, max_file_size, min_quality):
    """
    使用 Pillow 保存图像，设置 DPI，控制压缩质量和文件大小。
    """
    try:
        # 将OpenCV的BGR图像转换为RGB
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        # 准备保存的参数
        quality = 95  # 初始高质量
        while quality >= min_quality:
            # 保存到字节缓冲区
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG', quality=quality, dpi=(target_dpi, target_dpi))
            size = buffer.tell()
            if size <= max_file_size or quality == min_quality:
                break
            quality -= 5  # 逐步降低质量

        if size > max_file_size:
            # 无法在最低质量下满足文件大小要求
            return False, f"无法在最低质量 ({min_quality}) 下将文件大小控制在 {max_file_size // 1024}KB 以内。当前大小: {size // 1024}KB。"

        # 写入文件
        with open(output_path, 'wb') as f:
            f.write(buffer.getvalue())

        return True, f"设置质量={quality}, 文件大小={size // 1024}KB"
    except Exception as e:
        return False, f"保存图像时出错: {e}"

def process_image(file_path, output_path, face_mesh, log_queue, handle_overexposure, threshold, proportion, stop_event, face_width_ratio, vertical_offset, output_width, output_height, target_dpi, max_file_size, min_quality, v_adjust):
    """
    处理单张图像，包括旋转、识别、缩放、平移、过曝检测与修复、裁剪和保存。
    """
    try:
        if stop_event.is_set():
            log_queue.put((f"处理中止: {file_path.name}\n", "info"))
            return

        # 确保 file_path 是字符串
        file_path_str = str(file_path)

        # 替代 cv2.imread 以支持中文路径
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            image_array = np.frombuffer(data, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if image is None:
                log_queue.put((f"无法解码图像: {file_path}\n", "error"))
                return
        except Exception as e:
            log_queue.put((f"无法读取图像: {file_path}\n", "error"))
            return

        # 1. 旋转图像以确保为竖向
        image = rotate_image_if_needed(image)

        height, width = image.shape[:2]

        # 2. 获取人脸关键点
        landmarks = get_face_landmarks(image, face_mesh)
        if landmarks is None:
            log_queue.put((f"未检测到人脸: {file_path.name}\n", "error"))
            return
        
        image = align_face(image, landmarks)
        height, width = image.shape[:2]
        landmarks = get_face_landmarks(image, face_mesh)
        if landmarks is None:
            log_queue.put((f"未检测到人脸: {file_path.name}\n", "error"))
            return
        
        # 3. 获取鼻子位置
        nose_pos = get_nose_position(landmarks, width, height)

        # 4. 获取脸部宽度
        face_width = get_face_width(landmarks, width, height)
        if face_width == 0:
            log_queue.put((f"无法计算脸部宽度: {file_path.name}\n", "error"))
            return

        if stop_event.is_set():
            log_queue.put((f"处理中止: {file_path.name}\n", "info"))
            return

        # 5. 缩放图像以满足脸部宽度的比例
        target_face_width = output_width * face_width_ratio  # 比例值
        resized_image, scale_factor = scale_image(image, face_width, target_face_width)

        # 6. 计算缩放后的鼻子位置
        scaled_nose_pos = (int(nose_pos[0] * scale_factor), int(nose_pos[1] * scale_factor))

        if stop_event.is_set():
            log_queue.put((f"处理中止: {file_path.name}\n", "info"))
            return

        # 7. 平移图像以调整鼻子位置
        desired_nose_pos = (output_width // 2, output_height // 2 + vertical_offset)
        translated_image = translate_image(resized_image, scaled_nose_pos, desired_nose_pos, output_width, output_height)

        if stop_event.is_set():
            log_queue.put((f"处理中止: {file_path.name}\n", "info"))
            return

        # 8. 裁剪或填充图像到指定尺寸
        final_image = crop_and_pad(translated_image, output_width, output_height)

        if stop_event.is_set():
            log_queue.put((f"处理中止: {file_path.name}\n", "info"))
            return

        # 9. 检测并修复过曝（根据用户选择）
        if handle_overexposure:
            if is_overexposed(final_image, threshold=threshold, proportion=proportion):
                log_queue.put((f"{output_path.name} - 检测到过曝，正在修复...\n", "info"))
                final_image = fix_overexposure(final_image, v_adjust)
                log_queue.put((f"{output_path.name} - 过曝修复完成。\n", "info"))
            else:
                log_queue.put((f"{output_path.name} - 未检测到过曝。\n", "info"))
        else:
            log_queue.put((f"{output_path.name} - 过曝处理已禁用。\n", "info"))

        if stop_event.is_set():
            log_queue.put((f"处理中止: {file_path.name}\n", "info"))
            return

        # 10. 保存图像，设置 DPI 和压缩质量
        output_path_str = str(output_path)
        success, message = save_image_with_constraints(final_image, output_path_str, target_dpi, max_file_size, min_quality)
        if success:
            log_queue.put((f"{output_path.name} - 已保存: {message}\n", "info"))
        else:
            log_queue.put((f"{output_path.name} - {message}\n", "error"))

        # 11. 添加一个空行以提高日志可读性
        log_queue.put(("\n", "info"))

    except Exception as e:
        error_message = traceback.format_exc()
        log_queue.put((f"处理 {file_path.name} 时出错: {e}\n{error_message}\n", "error"))
        # 即使出错，也添加一个空行
        log_queue.put(("\n", "info"))

def select_directory():
    """
    打开文件夹选择对话框，让用户选择要处理的目录。
    """
    directory = filedialog.askdirectory(title="选择文件所在目录")
    if directory:
        # 使用绝对路径并转换为 Path 对象
        dir_var.set(str(Path(directory).resolve()))

def start_processing():
    """
    开始批量处理图像，运行在单独的线程中。
    """
    global processing_thread_instance, stop_event, completion_shown
    global OUTPUT_WIDTH, OUTPUT_HEIGHT, TARGET_DPI, MAX_FILE_SIZE, MIN_QUALITY, V_CHANNEL_ADJUST
    global VALID_EXTENSIONS

    # 初始化完成提示标志
    if not hasattr(start_processing, "completion_shown"):
        start_processing.completion_shown = False
    start_processing.completion_shown = False  # 每次开始处理前重置标志

    if not processing_thread_instance or not processing_thread_instance.is_alive():
        # 开始处理
        input_dir_str = dir_var.get()
        if not input_dir_str:
            messagebox.showwarning("警告", "请选择一个目录！")
            return

        input_dir = Path(input_dir_str)
        if not input_dir.exists():
            messagebox.showerror("错误", f"选择的目录不存在: {input_dir}")
            return

        output_dir = input_dir / 'CUT'
        if not output_dir.exists():
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                messagebox.showerror("错误", f"无法创建输出目录: {output_dir}\n{e}")
                return

        # 获取用户是否选择处理过曝
        handle_overexposure = overexposure_var.get()

        # 获取用户设置的亮度阈值和比例阈值
        try:
            threshold = int(entry_threshold.get())
            if not (0 <= threshold <= 255):
                raise ValueError
        except ValueError:
            messagebox.showerror("错误", "亮度阈值必须为0到255之间的整数。")
            return

        try:
            proportion = float(entry_proportion.get()) / 100  # 将百分比转换为比例
            if not (0.0 <= proportion <= 1.0):
                raise ValueError
        except ValueError:
            messagebox.showerror("错误", "亮度比例阈值必须为0到100之间的数值。")
            return

        # 获取用户设置的脸宽比例和垂直位移
        try:
            face_width_ratio = float(entry_face_width_ratio.get())
            if not (0.0 <= face_width_ratio <= 1.0):
                raise ValueError
        except ValueError:
            messagebox.showerror("错误", "脸宽比例必须为0到1之间的数值。")
            return

        try:
            vertical_offset = int(entry_vertical_offset.get())
            if not (- OUTPUT_HEIGHT.get() <= vertical_offset <= OUTPUT_HEIGHT.get()):
                raise ValueError
        except ValueError:
            messagebox.showerror("错误", "垂直位移必须为正负图像高度之间的整数。")
            return

        # 获取用户设置的输出图像尺寸、DPI、文件大小和最低压缩质量
        try:
            output_width = int(entry_output_width.get())
            output_height = int(entry_output_height.get())
            if output_width <= 0 or output_height <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("错误", "输出图像宽度和高度必须为正整数。")
            return

        try:
            target_dpi = int(entry_target_dpi.get())
            if target_dpi <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("错误", "DPI 必须为正整数。")
            return

        try:
            max_file_size = int(entry_max_file_size.get()) * 1024  # 将 KB 转换为字节
            if max_file_size <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("错误", "文件大小必须为正整数KB。")
            return

        try:
            min_quality = int(entry_min_quality.get())
            if not (0 <= min_quality <= 100):
                raise ValueError
        except ValueError:
            messagebox.showerror("错误", "最低压缩质量必须为0到100之间的整数。")
            return

        try:
            v_adjust = float(entry_v_channel_adjust.get())
            if not (0.0 <= v_adjust <= 1.0):
                raise ValueError
        except ValueError:
            messagebox.showerror("错误", "V 通道调整系数必须为0到1之间的数值。")
            return

        # 创建日志队列
        log_queue = queue.Queue()

        # 创建停止事件
        stop_event = threading.Event()

        # 实例化处理线程
        def processing_thread_func():
            try:
                with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True,
                                                   min_detection_confidence=0.5) as face_mesh:
                    # 遍历输入目录下所有文件
                    for file_path in input_dir.iterdir():
                        if stop_event.is_set():
                            log_queue.put(("操作已停止。\n", "info"))
                            break
                        if file_path.suffix.lower() in VALID_EXTENSIONS:
                            output_filename = file_path.stem + ".jpg"  # 保存为JPEG格式
                            output_path = output_dir / output_filename
                            log_queue.put((f"处理图像: {file_path.name}\n", "info"))
                            process_image(file_path, output_path, face_mesh, log_queue, handle_overexposure, threshold, proportion, stop_event, face_width_ratio, vertical_offset, output_width, output_height, target_dpi, max_file_size, min_quality, v_adjust)
                            log_queue.put(("__PROGRESS_UPDATE__", "progress"))
            except Exception as e:
                error_message = traceback.format_exc()
                log_queue.put((f"批量处理时出错: {e}\n{error_message}\n", "error"))
            finally:
                log_queue.put(("__DONE__", None))

        processing_thread_instance = threading.Thread(target=processing_thread_func, daemon=True)
        processing_thread_instance.start()

        # 更新按钮状态为“停止”
        btn_confirm.config(text="停止", bg=root.cget('bg'))

        # 初始化进度条
        total_files = len([f for f in input_dir.iterdir() if f.suffix.lower() in VALID_EXTENSIONS])
        if total_files == 0:
            messagebox.showinfo("信息", "选择的目录中没有支持的图片文件。")
            btn_confirm.config(text="确认并开始处理")
            return

        progress_bar['value'] = 0
        progress_bar['maximum'] = total_files

        log_widget.config(state=tk.NORMAL)
        log_widget.delete(1.0, tk.END)
        log_widget.insert(tk.END, f"开始处理目录: {input_dir}\n")
        log_widget.config(state=tk.DISABLED)

        def update_logs():
            try:
                while True:
                    message, msg_type = log_queue.get_nowait()
                    if message == "__PROGRESS_UPDATE__":
                        progress_bar['value'] += 1
                    elif message == "__DONE__":
                        if stop_event.is_set():
                            log_widget.config(state=tk.NORMAL)
                            log_widget.insert(tk.END, "处理已停止。\n", "info")
                            log_widget.config(state=tk.DISABLED)
                            if not start_processing.completion_shown:
                                start_processing.completion_shown = True
                        else:
                            log_widget.config(state=tk.NORMAL)
                            log_widget.insert(tk.END, "所有图像已处理完成！\n", "info")
                            log_widget.config(state=tk.DISABLED)
                            if not start_processing.completion_shown:
                                start_processing.completion_shown = True
                        # 重置按钮
                        btn_confirm.config(text="确认并开始处理", bg=root.cget('bg'))
                    else:
                        log_widget.config(state=tk.NORMAL)
                        if msg_type == "error":
                            log_widget.insert(tk.END, message, "error")
                        elif msg_type == "info":
                            log_widget.insert(tk.END, message, "info")
                        else:
                            log_widget.insert(tk.END, message)
                        log_widget.see(tk.END)
                        log_widget.config(state=tk.DISABLED)
            except queue.Empty:
                pass
            if processing_thread_instance.is_alive() or not log_queue.empty():
                root.after(100, update_logs)
            else:
                if not start_processing.completion_shown:
                    log_widget.config(state=tk.NORMAL)
                    log_widget.insert(tk.END, "所有图像已处理完成！\n", "info")
                    log_widget.config(state=tk.DISABLED)
                    start_processing.completion_shown = True
                # 重置按钮，如果按钮还在“停止”状态
                if btn_confirm['text'] == "停止":
                    btn_confirm.config(text="确认并开始处理", bg=root.cget('bg'))

        root.after(100, update_logs)

    elif processing_thread_instance and processing_thread_instance.is_alive():
        # 停止处理
        stop_event.set()
        btn_confirm.config(state=tk.NORMAL)

def main():
    try:
        # 创建 GUI
        global root, dir_var, log_widget, btn_confirm, progress_bar, overexposure_var
        global entry_threshold, entry_proportion, entry_face_width_ratio, entry_vertical_offset
        global entry_output_width, entry_output_height, entry_target_dpi, entry_max_file_size, entry_min_quality, entry_v_channel_adjust
        global processing_thread_instance, stop_event
        global OUTPUT_WIDTH, OUTPUT_HEIGHT, TARGET_DPI, MAX_FILE_SIZE, MIN_QUALITY, V_CHANNEL_ADJUST

        root = tk.Tk()
        root.title("人脸批量处理程序 v0.95 -- Powered By Shining Hu -- 2024/09/25")
        root.geometry("600x655")  # 增大窗口尺寸以适应更多组件

        # 初始默认值
        DEFAULT_WIDTH = 480
        DEFAULT_HEIGHT = 640
        DEFAULT_DPI = 300
        DEFAULT_MAX_FILE_SIZE = 40  # 40KB
        DEFAULT_MIN_QUALITY = 60
        DEFAULT_V_CHANNEL_ADJUST = 0.9

        # 全局变量
        OUTPUT_WIDTH = tk.IntVar(value=DEFAULT_WIDTH)
        OUTPUT_HEIGHT = tk.IntVar(value=DEFAULT_HEIGHT)
        TARGET_DPI = tk.IntVar(value=DEFAULT_DPI)
        MAX_FILE_SIZE = tk.IntVar(value=DEFAULT_MAX_FILE_SIZE)
        MIN_QUALITY = tk.IntVar(value=DEFAULT_MIN_QUALITY)
        V_CHANNEL_ADJUST = tk.DoubleVar(value=DEFAULT_V_CHANNEL_ADJUST)

        # 选择目录部分
        frame_select = tk.Frame(root)
        frame_select.pack(pady=20)

        dir_var = tk.StringVar()

        btn_browse = tk.Button(frame_select, text="选择照片目录", command=select_directory)
        btn_browse.pack(side=tk.LEFT, padx=20)

        entry_dir = tk.Entry(frame_select, textvariable=dir_var, width=60)
        entry_dir.pack(side=tk.LEFT, padx=5)

        # 输出图像尺寸
        frame_output_size = tk.Frame(root)
        frame_output_size.pack(pady=5)

        lbl_output_width = tk.Label(frame_output_size, text="图像宽度:")
        lbl_output_width.pack(side=tk.LEFT, padx=5)

        entry_output_width = tk.Entry(frame_output_size, textvariable=OUTPUT_WIDTH, width=5)
        entry_output_width.pack(side=tk.LEFT, padx=5)

        lbl_output_height = tk.Label(frame_output_size, text="图像高度:")
        lbl_output_height.pack(side=tk.LEFT, padx=5)

        entry_output_height = tk.Entry(frame_output_size, textvariable=OUTPUT_HEIGHT, width=5)
        entry_output_height.pack(side=tk.LEFT, padx=5)

        # DPI 、文件大小限制、最低压缩质量 设置
        frame_dpi = tk.Frame(root)
        frame_dpi.pack(pady=5)

        lbl_target_dpi = tk.Label(frame_dpi, text="DPI:")
        lbl_target_dpi.pack(side=tk.LEFT, padx=5)

        entry_target_dpi = tk.Entry(frame_dpi, textvariable=TARGET_DPI, width=5)
        entry_target_dpi.pack(side=tk.LEFT, padx=4)

        lbl_max_file_size = tk.Label(frame_dpi, text="文件最大 (KB):")
        lbl_max_file_size.pack(side=tk.LEFT, padx=5)

        entry_max_file_size = tk.Entry(frame_dpi, textvariable=MAX_FILE_SIZE, width=5)
        entry_max_file_size.pack(side=tk.LEFT, padx=5)

        lbl_min_quality = tk.Label(frame_dpi, text="最低质量 (0-100):")
        lbl_min_quality.pack(side=tk.LEFT, padx=5)

        entry_min_quality = tk.Entry(frame_dpi, textvariable=MIN_QUALITY, width=3)
        entry_min_quality.pack(side=tk.LEFT, padx=5)

        # 脸宽比例
        frame_face = tk.Frame(root)
        frame_face.pack(pady=5)

        lbl_face_width_ratio = tk.Label(frame_face, text="脸宽比例 (0-1):")
        lbl_face_width_ratio.pack(side=tk.LEFT, padx=5)

        entry_face_width_ratio = tk.Entry(frame_face, width=5)
        entry_face_width_ratio.delete(0, tk.END)
        entry_face_width_ratio.insert(0, "0.52")  # 默认值为0.52
        entry_face_width_ratio.pack(side=tk.LEFT, padx=5)

        # 垂直位移
        lbl_vertical_offset = tk.Label(frame_face, text="脸中心垂直位移:")
        lbl_vertical_offset.pack(side=tk.LEFT, padx=5)

        entry_vertical_offset = tk.Entry(frame_face, width=4)
        entry_vertical_offset.delete(0, tk.END)
        entry_vertical_offset.insert(0, "22")  # 默认值为22
        entry_vertical_offset.pack(side=tk.LEFT, padx=5)

        frame_kongbai = tk.Frame(root)
        frame_kongbai.pack(pady=5)

        # 过曝处理选项
        frame_overexposure = tk.Frame(root)
        frame_overexposure.pack(pady=5)

        overexposure_var = tk.BooleanVar(value=False)  # 默认不选中， 不处理过曝

        chk_overexposure = tk.Checkbutton(
            frame_overexposure,
            text="处理过曝（同时满足高亮阈值和亮度比例阈值的照片将被减亮）",
            variable=overexposure_var
        )
        chk_overexposure.pack()

        # 添加亮度阈值和比例阈值的界面控件
        frame_threshold = tk.Frame(root)
        frame_threshold.pack(pady=5)

        # 亮度阈值
        lbl_threshold = tk.Label(frame_threshold, text="高亮阈值 (0-255):")
        lbl_threshold.pack(side=tk.LEFT, padx=5)

        entry_threshold = tk.Entry(frame_threshold, width=5)
        entry_threshold.delete(0, tk.END)
        entry_threshold.insert(0, "240")  # 默认值
        entry_threshold.pack(side=tk.LEFT, padx=5)

        # 亮度比例阈值
        lbl_proportion = tk.Label(frame_threshold, text="亮度比例阈值 (%):")
        lbl_proportion.pack(side=tk.LEFT, padx=5)

        entry_proportion = tk.Entry(frame_threshold, width=3)
        entry_proportion.delete(0, tk.END)
        entry_proportion.insert(0, "10")  # 默认值为10%
        entry_proportion.pack(side=tk.LEFT, padx=5)

        # V 通道调整系数
        lbl_v_channel_adjust = tk.Label(frame_threshold, text="减亮系数 (0.0-1.0):")
        lbl_v_channel_adjust.pack(side=tk.LEFT, padx=5)

        entry_v_channel_adjust = tk.Entry(frame_threshold, textvariable=V_CHANNEL_ADJUST, width=3)
        entry_v_channel_adjust.pack(side=tk.LEFT, padx=5)

        # 确认按钮
        btn_confirm = tk.Button(root, text="确认并开始处理", command=start_processing, width=20)
        btn_confirm.pack(pady=10)

        # 进度条
        progress_bar = ttk.Progressbar(root, orient='horizontal', length=570, mode='determinate')
        progress_bar.pack(pady=10)

        # 日志显示区域
        log_widget = scrolledtext.ScrolledText(root, width=80, height=20, state=tk.DISABLED)
        log_widget.pack(padx=10, pady=10)

        # 定义日志消息的标签
        log_widget.tag_configure("error", foreground="red")
        log_widget.tag_configure("info", foreground="black")  # 默认颜色


        # 添加底部信息
        frame_buttommess = tk.Frame(root)
        frame_buttommess.pack(pady=0)
        buttommessage = tk.Label(frame_buttommess, text="(C)opyright 2024-2024 - Powered By Shining Hu")
        buttommessage.pack(side=tk.LEFT, padx=5)


        # 初始化处理线程实例
        processing_thread_instance = None
        stop_event = None

        # 运行 GUI 主循环
        root.mainloop()
    except Exception as e:
        error_log = Path('error_log.txt')
        error_log.write_text(traceback.format_exc(), encoding='utf-8')
        messagebox.showerror("未处理的异常", f"程序运行时出错，请查看 {error_log.resolve()} 了解详情。\n{e}")

if __name__ == "__main__":
    main()

