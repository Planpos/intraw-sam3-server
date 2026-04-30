import base64
import copy
import io
import logging
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 자동 탐지 시 사용할 카테고리 목록
AUTO_CATEGORIES = [
    "person", "car", "truck", "bus", "motorcycle", "bicycle",
    "dog", "cat", "bird", "horse", "cow", "sheep",
    "chair", "couch", "table", "bed", "desk",
    "tv", "laptop", "phone", "keyboard", "mouse",
    "bottle", "cup", "bowl", "plate", "fork", "knife",
    "backpack", "bag", "suitcase", "umbrella", "hat",
    "book", "clock", "vase", "plant", "flower",
    "door", "window", "wall", "floor", "ceiling",
    "ball", "bat", "racket","outer wall","carpet",
    "pizza", "cake", "sandwich", "apple", "banana",
]

LABEL_KO = {
    "person": "사람", "car": "자동차", "truck": "트럭", "bus": "버스",
    "motorcycle": "오토바이", "bicycle": "자전거",
    "dog": "강아지", "cat": "고양이", "bird": "새", "horse": "말",
    "cow": "소", "sheep": "양",
    "chair": "의자", "couch": "소파", "table": "테이블", "bed": "침대", "desk": "책상",
    "tv": "TV", "laptop": "노트북", "phone": "휴대폰", "keyboard": "키보드", "mouse": "마우스",
    "bottle": "병", "cup": "컵", "bowl": "그릇", "plate": "접시",
    "fork": "포크", "knife": "칼",
    "backpack": "백팩", "bag": "가방", "suitcase": "여행가방",
    "umbrella": "우산", "hat": "모자",
    "book": "책", "clock": "시계", "vase": "꽃병", "plant": "식물", "flower": "꽃",
    "door": "문", "window": "창문", "wall": "벽", "floor": "바닥", "ceiling": "천장",
    "ball": "공", "bat": "배트", "racket": "라켓", "outer wall": "외벽", "carpet": "카펫",
    "pizza": "피자", "cake": "케이크", "sandwich": "샌드위치",
    "apple": "사과", "banana": "바나나",
}

processor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global processor
    logger.info("SAM3 모델 로딩 중...")
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam3_image_model()
    processor = Sam3Processor(model, device=device)
    logger.info("SAM3 모델 로딩 완료")
    yield
    del processor


app = FastAPI(title="SAM3 Image Analysis Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def mask_to_base64(mask: np.ndarray) -> str:
    img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def box_iou(box1, box2):
    """두 박스의 IoU 계산 (x1,y1,x2,y2 형식)"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0


def nms_detections(detections, iou_threshold=0.5):
    """중복 박스 제거 (NMS) - 같은 레이블 내에서만 적용"""
    if not detections:
        return []
    detections = sorted(detections, key=lambda x: x["score"], reverse=True)
    kept = []
    for det in detections:
        b1 = [det["box"]["x1"], det["box"]["y1"], det["box"]["x2"], det["box"]["y2"]]
        overlap = any(
            k["label"] == det["label"] and
            box_iou(b1, [k["box"]["x1"], k["box"]["y1"], k["box"]["x2"], k["box"]["y2"]]) > iou_threshold
            for k in kept
        )
        if not overlap:
            kept.append(det)
    return kept


def run_inference_with_state(state, prompt, confidence_threshold):
    """인코딩된 state로 단일 프롬프트 추론"""
    try:
        import copy
        processor.set_confidence_threshold(confidence_threshold)
        s = processor.set_text_prompt(prompt=prompt, state=copy.deepcopy(state))

        boxes = s["boxes"].cpu().tolist()
        scores = s["scores"].cpu().tolist()
        masks = s["masks"].cpu().squeeze(1).numpy()

        if scores:
            logger.info(f"[{prompt}] {len(scores)}개 감지, 최고 score: {max(scores):.3f}")

        detections = []
        for box, score, mask in zip(boxes, scores, masks):
            detections.append({
                "label": LABEL_KO.get(prompt, prompt),
                "score": round(score, 4),
                "box": {"x1": round(box[0], 2), "y1": round(box[1], 2),
                        "x2": round(box[2], 2), "y2": round(box[3], 2)},
                "mask_base64": mask_to_base64(mask),
            })
        return detections
    except Exception as e:
        logger.warning(f"[{prompt}] 추론 실패: {e}")
        return []


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": processor is not None}


@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    prompt: str = Form(default=""),
    confidence_threshold: float = Form(default=0.4),
):
    if processor is None:
        raise HTTPException(status_code=503, detail="모델 로딩 중입니다.")

    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 파일 오류: {e}")

    orig_w, orig_h = pil_image.size
    auto_mode = prompt.strip() in ("", "object", "auto")
    logger.info(f"prompt='{prompt}' auto_mode={auto_mode} threshold={confidence_threshold}")

    try:
        if auto_mode:
            # 자동 모드: 이미지 인코딩 1회 → 카테고리별 text prompt만 교체
            logger.info("자동 탐지 모드 실행 중...")
            base_state = processor.set_image(pil_image)
            all_detections = []
            for category in AUTO_CATEGORIES:
                dets = run_inference_with_state(base_state, category, confidence_threshold)
                all_detections.extend(dets)
            detections = nms_detections(all_detections, iou_threshold=0.5)
            logger.info(f"NMS 전 {len(all_detections)}개 → NMS 후 {len(detections)}개")
            used_prompt = "auto"
        else:
            # 수동 모드: 입력된 프롬프트로 탐지
            base_state = processor.set_image(pil_image)
            detections = run_inference_with_state(base_state, prompt.strip(), confidence_threshold)
            used_prompt = prompt.strip()
    except Exception as e:
        logger.exception("추론 오류")
        raise HTTPException(status_code=500, detail=f"추론 오류: {e}")

    # id 재부여
    for i, det in enumerate(detections):
        det["id"] = i

    return {
        "image_width": orig_w,
        "image_height": orig_h,
        "prompt": used_prompt,
        "auto_mode": auto_mode,
        "num_detections": len(detections),
        "detections": detections,
    }
