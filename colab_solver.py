# colab_solver.py — يعمل في Google Colab
# يستقبل الصورة من window.pending_image ويكتب النتيجة في window.yolo_result

import asyncio
import base64
import json
import numpy as np
import cv2
from ultralytics import YOLO
import websockets

# ═══════════════════════════════════════
# إعدادات
# ═══════════════════════════════════════
MODEL_PATH   = "yolov8n-seg.pt"   # ملف الوزن
RAILWAY_WS   = "wss://lab-automation-production-1897.up.railway.app/ws"                 # ← ضع هنا رابط WebSocket Railway
CONF         = 0.05               # حد الثقة

# خريطة الأسماء العربية/الإنجليزية → class index في YOLO
CAPTCHA_CLASSES = {
    'car': 2, 'cars': 2,
    'bus': 5, 'buses': 5,
    'motorcycle': 3, 'motorcycles': 3,
    'bicycle': 1, 'bicycles': 1,
    'traffic light': 9, 'traffic lights': 9,
    'fire hydrant': 10, 'fire hydrants': 10,
    'boat': 8, 'boats': 8,
    'truck': 7, 'trucks': 7,
}

# تقسيم الشبكة: 3×3 أو 4×4
GRID = {
    9:  {'rows': 3, 'cols': 3},
    16: {'rows': 4, 'cols': 4},
}

print("⏳ تحميل YOLO...")
model = YOLO(MODEL_PATH)
print("✅ YOLO جاهز!")

# ═══════════════════════════════════════
# دالة الحل الرئيسية
# ═══════════════════════════════════════
def solve_captcha_image(img_bytes: bytes, question: str, grid_size: int) -> list:
    """
    تحلّل الصورة وتُرجع أرقام الخلايا [1,4,7]
    """
    # تحويل bytes → numpy
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return []

    h, w = img.shape[:2]
    grid = GRID.get(grid_size, GRID[9])
    rows, cols = grid['rows'], grid['cols']
    cell_w = w / cols
    cell_h = h / rows

    # اكتشف class index من السؤال
    question_lower = question.lower()
    class_idx = None
    for keyword, idx in CAPTCHA_CLASSES.items():
        if keyword in question_lower:
            class_idx = idx
            break

    if class_idx is None:
        print(f"⚠️ لم أجد class لـ: {question}")
        return []

    # شغّل YOLO
    results = model.predict(img, conf=CONF, verbose=False)
    if not results or results[0].masks is None:
        print("⚠️ YOLO: لا يوجد masks")
        return []

    masks     = results[0].masks.data.cpu().numpy()
    classes   = results[0].boxes.cls.cpu().int().numpy()

    # فلتر: فقط الـ masks التي تطابق class_idx
    target_masks = masks[classes == class_idx]
    if not target_masks.size:
        print(f"⚠️ لم يجد {question} في الصورة")
        return []

    # لكل mask — اكتشف أي خلايا تتقاطع معها
    selected_cells = set()
    for mask in target_masks:
        # resize الـ mask لتطابق حجم الصورة
        mask_resized = cv2.resize(mask, (w, h))
        for r in range(rows):
            for c in range(cols):
                cell_num = r * cols + c + 1
                x1 = int(c * cell_w)
                y1 = int(r * cell_h)
                x2 = int((c + 1) * cell_w)
                y2 = int((r + 1) * cell_h)
                # تحقق هل الـ mask تتقاطع مع هذه الخلية
                cell_mask = mask_resized[y1:y2, x1:x2]
                overlap = cv2.countNonZero(cell_mask)
                # عتبة التقاطع حسب نوع الشبكة
                threshold = 3 if grid_size == 9 else 8
                if overlap > threshold:
                    selected_cells.add(cell_num)

    result = sorted(selected_cells)
    print(f"✅ YOLO نتيجة: {result} | سؤال: {question[:40]}")
    return result


# ═══════════════════════════════════════
# WebSocket heartbeat لـ Railway
# ═══════════════════════════════════════
async def send_heartbeat():
    """يرسل إشارة كل ثانية إلى Railway"""
    while True:
        try:
            async with websockets.connect(RAILWAY_WS, ping_interval=None) as ws:
                print(f"✅ متصل بـ Railway WebSocket")
                while True:
                    await ws.send("heartbeat")
                    await asyncio.sleep(1)
        except Exception as e:
            print(f"⚠️ WebSocket انقطع: {e} — إعادة محاولة بعد 3 ثوانٍ")
            await asyncio.sleep(3)


# ═══════════════════════════════════════
# الحلقة الرئيسية — يراقب window variables
# ═══════════════════════════════════════
# هذا الكود يُشغَّل من Colab cell عبر:
#   from colab_solver import solve_captcha_image, send_heartbeat
#   asyncio.create_task(send_heartbeat())

print("✅ colab_solver.py محمّل وجاهز!")
