import sys
import torch
from PIL import Image, ImageDraw
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# 테스트용 이미지 경로를 인자로 받거나 간단한 합성 이미지 사용
img_path = sys.argv[1] if len(sys.argv) > 1 else None

if img_path:
    img = Image.open(img_path).convert("RGB")
    print(f"이미지 로드: {img_path} ({img.size})")
else:
    # 합성 이미지: 밝은 사각형 2개
    img = Image.new("RGB", (800, 600), (200, 200, 200))
    draw = ImageDraw.Draw(img)
    draw.rectangle([100, 100, 350, 450], fill=(50, 100, 200))
    draw.rectangle([450, 150, 700, 500], fill=(200, 80, 80))
    print("합성 테스트 이미지 사용 (800x600)")

print("모델 로딩 중...")
m = build_sam3_image_model()
p = Sam3Processor(m, device="cuda", confidence_threshold=0.0)  # threshold=0으로 전부 출력

print("이미지 인코딩 중...")
state = p.set_image(img)

prompts = ["object", "thing", "item", "region"]
for prompt in prompts:
    import copy
    s = p.set_text_prompt(prompt, copy.deepcopy(state))
    scores = sorted(s["scores"].tolist(), reverse=True)
    top5 = [round(v, 3) for v in scores[:5]]
    print(f"[{prompt:10s}] 총 {len(scores)}개 | 상위 score: {top5}")
