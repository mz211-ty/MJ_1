"""
粉色心形素描海报生成器
输入一张照片，输出 1920x1080 的 16:9 海报：
  - 经典 color-dodge 铅笔素描线稿
  - 线稿着粉色
  - 心形遮罩裁切
  - 右下角加文字「爱人的眼睛」
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageChops, ImageFont

# ---------- 可调参数 ----------
CANVAS_W, CANVAS_H = 1920, 1080
SUBJECT_SIZE = 720          # 心形主体在画布上的最大边长
BLUR_RADIUS = 8             # 高斯模糊半径（越大线条越粗、留白越多）
EDGE_ENHANCE = 1.4          # 边缘增强强度，1.0 为不增强
PINK = (255, 110, 150)      # 线条粉色
TEXT = "爱人的眼睛"
FONT_SIZE = 96
# -----------------------------


def _find_chinese_font():
    """跨平台查找一个可用的中文字体，找不到返回 None。"""
    candidates = [
        # macOS
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        # Linux
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        # Windows
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/msyh.ttc",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def _color_dodge(gray, radius):
    """
    经典铅笔素描：反色 → 高斯模糊 → 颜色减淡。
    公式: dodge = gray * 256 / (256 - blurred)，分母做保护。
    """
    arr = np.asarray(gray, dtype=np.float32)
    # 反色
    inv = 255.0 - arr
    inv_img = Image.fromarray(inv.astype(np.uint8))
    # 高斯模糊
    blurred = np.asarray(
        inv_img.filter(ImageFilter.GaussianBlur(radius=radius)),
        dtype=np.float32,
    )
    # 颜色减淡，分母保护
    denom = 256.0 - blurred
    denom = np.where(denom < 1.0, 1.0, denom)
    dodge = (arr * 256.0) / denom
    dodge = np.clip(dodge, 0, 255).astype(np.uint8)
    return Image.fromarray(dodge, mode="L")


def _enhance_edges(sketch, strength):
    """轻微提升线条对比度，让黑线更黑、白底更白。"""
    arr = np.asarray(sketch, dtype=np.float32)
    # 以 255 为中心拉伸：离 255 越远（线条越黑），按 strength 放大差距
    dist = 255.0 - arr
    arr = 255.0 - np.clip(dist * strength, 0, 255)
    return Image.fromarray(arr.astype(np.uint8), mode="L")


def _fit_square(img, size):
    """等比缩放并居中裁剪到 size×size，避免变形。"""
    w, h = img.size
    scale = max(size / w, size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - size) // 2
    top = (new_h - size) // 2
    return img.crop((left, top, left + size, top + size))


def _heart_mask(size):
    """生成 size×size 的心形蒙版（白色=保留，黑色=透明）。"""
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    # 用参数方程画一个饱满的心形
    cx, cy = size / 2, size * 0.54
    scale = size / 34.0
    points = []
    for t in np.linspace(0, 2 * np.pi, 400):
        x = 16 * np.sin(t) ** 3
        y = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
        points.append((cx + x * scale, cy + y * scale))
    draw.polygon(points, fill=255)
    # 边缘轻微羽化，避免生硬锯齿
    return mask.filter(ImageFilter.GaussianBlur(radius=1.5))


def _tint_pink(sketch, pink):
    """把灰度素描的黑线着成粉色，白底保持透明。"""
    sketch_arr = np.asarray(sketch)  # (S, S) uint8，越黑值越小
    pink_rgb = np.array(pink, dtype=np.uint8)
    # alpha = 线条深浅：黑线 alpha≈255，白底 alpha≈0
    alpha = (255 - sketch_arr).astype(np.uint8)
    rgba = np.zeros((*sketch_arr.shape, 4), dtype=np.uint8)
    rgba[..., :3] = pink_rgb
    rgba[..., 3] = alpha
    return Image.fromarray(rgba, mode="RGBA")


def create_styled_image(input_path, output_path, text=TEXT):
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), (255, 255, 255))

    # 1. 读取并等比缩放成方形
    try:
        src = Image.open(input_path)
    except Exception as e:
        print(f"无法打开图片 {input_path}: {e}")
        return

    src_square = _fit_square(src.convert("RGB"), SUBJECT_SIZE)

    # 2. color-dodge 素描 + 边缘增强
    gray = src_square.convert("L")
    sketch = _color_dodge(gray, BLUR_RADIUS)
    sketch = _enhance_edges(sketch, EDGE_ENHANCE)

    # 3. 着粉色
    tinted = _tint_pink(sketch, PINK)

    # 4. 心形遮罩
    heart = _heart_mask(SUBJECT_SIZE)
    rgba_heart = Image.new("RGBA", (SUBJECT_SIZE, SUBJECT_SIZE), (0, 0, 0, 0))
    rgba_heart.paste(tinted, (0, 0), heart)

    # 5. 居中贴到画布（微上移）
    paste_x = (CANVAS_W - SUBJECT_SIZE) // 2
    paste_y = (CANVAS_H - SUBJECT_SIZE) // 2 - 70
    canvas.paste(rgba_heart, (paste_x, paste_y), rgba_heart)

    # 6. 右下角文字
    font_path = _find_chinese_font()
    if font_path:
        font = ImageFont.truetype(font_path, FONT_SIZE)
        draw = ImageDraw.Draw(canvas)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        tx = CANVAS_W - text_w - 120
        ty = CANVAS_H - text_h - 110
        # 文字投影，增加可读性
        draw.text((tx + 3, ty + 3), text, fill=(220, 200, 210), font=font)
        draw.text((tx, ty), text, fill=PINK, font=font)
    else:
        print("未找到中文字体，已跳过文字。可把字体路径加入 _find_chinese_font()。")

    canvas.save(output_path)
    print(f"处理完成: {output_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("用法: python styled_image.py <输入图> <输出图>")
    else:
        create_styled_image(sys.argv[1], sys.argv[2])
