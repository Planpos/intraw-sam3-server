import base64
import copy
import io
import json
import logging
import os
from contextlib import asynccontextmanager

import numpy as np
import torch
from scipy import ndimage
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── 카테고리 (auto_categories.json 에서 로드) ─────────────────────────────────
AUTO_CATEGORIES_FILE = os.path.join(os.path.dirname(__file__), "auto_categories.json")

def _load_auto_categories() -> tuple[list[str], dict[str, str]]:
    try:
        with open(AUTO_CATEGORIES_FILE, encoding="utf-8") as f:
            data = json.load(f)
        categories = [t for t in data.get("categories", []) if isinstance(t, str)]
        labels_ko  = {k: v for k, v in data.get("labels_ko", {}).items()}
        logger.info(f"auto_categories.json 로드 완료: {len(categories)}개")
        return categories, labels_ko
    except Exception as e:
        logger.error(f"auto_categories.json 로드 실패: {e} — 빈 목록으로 시작합니다.")
        return [], {}

AUTO_CATEGORIES, LABEL_KO = _load_auto_categories()

processor = None
interactive_predictor = None  # SAM2 방식 포인트 세그먼테이션용


@asynccontextmanager
async def lifespan(app: FastAPI):
    global processor, interactive_predictor
    logger.info("SAM3 모델 로딩 중...")
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam3_image_model(enable_inst_interactivity=True)
    processor = Sam3Processor(model, device=device)
    interactive_predictor = model.inst_interactive_predictor
    interactive_predictor.eval()
    logger.info(f"SAM3 모델 로딩 완료 | 카테고리 {len(AUTO_CATEGORIES)}개 | 포인트 세그먼테이션 활성화")
    yield
    del processor
    del interactive_predictor


app = FastAPI(title="SAM3 Image Analysis Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """가장 큰 연결 컴포넌트만 남기고 내부 구멍을 채운다."""
    labeled, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    largest = int(np.argmax(sizes)) + 1
    result = (labeled == largest)
    result = ndimage.binary_fill_holes(result)
    return result.astype(bool)


def mask_to_base64(mask: np.ndarray) -> str:
    img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def box_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0


def nms_detections(detections, iou_threshold=0.2):
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
    try:
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


@app.get("/demo")
def demo():
    return FileResponse(os.path.join(os.path.dirname(__file__), "demo.html"))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": processor is not None,
        "categories": len(AUTO_CATEGORIES),
    }


@app.get("/categories")
def list_categories():
    return {
        "categories": AUTO_CATEGORIES,
        "total": len(AUTO_CATEGORIES),
        "file": AUTO_CATEGORIES_FILE,
    }


def _setup_interactive_predictor(state, orig_h, orig_w):
    """
    processor.set_image()이 계산한 SAM2 백본 피처를
    interactive_predictor._features에 직접 주입한다.
    (tracker.backbone = None 이므로 set_image()를 직접 호출할 수 없음)
    """
    ip = interactive_predictor
    sam2_out = state["backbone_out"].get("sam2_backbone_out")
    if sam2_out is None:
        raise RuntimeError("sam2_backbone_out이 없습니다. enable_inst_interactivity=True 확인 필요")

    with torch.inference_mode():
        _, vision_feats, _, _ = ip.model._prepare_backbone_features(sam2_out)
        vision_feats[-1] = vision_feats[-1] + ip.model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], ip._bb_feat_sizes[::-1])
        ][::-1]

    ip._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
    ip._orig_hw = [(orig_h, orig_w)]
    ip._is_image_set = True
    ip._is_batch = False

@app.post("/segment-by-point")
async def segment_by_point(
    image: UploadFile = File(...),
    x: float = Form(...),
    y: float = Form(...),
):
    if interactive_predictor is None or processor is None:
        raise HTTPException(status_code=503, detail="모델 로딩 중입니다.")

    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 파일 오류: {e}")

    orig_w, orig_h = pil_image.size
    logger.info(f"포인트 세그멘테이션: pixel=({x},{y})")

    try:
        # processor로 이미지 인코딩 (SAM2 피처 포함)
        state = processor.set_image(pil_image)

        # SAM2 피처를 interactive_predictor에 주입
        _setup_interactive_predictor(state, orig_h, orig_w)

        point_coords = np.array([[x, y]], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)

        # multimask_output=True → 3개 후보 중 IOU 최고 선택
        masks, iou_scores, _ = interactive_predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=False,
        )

        # 클릭 픽셀을 포함하는 마스크 중 가장 넓은 것 선택
        # (IOU 최고값은 정밀한 부분만 잡는 경향 → 전체 객체가 잘 안 잡힘)
        px_i, py_i = int(round(x)), int(round(y))
        img_area = orig_w * orig_h
        candidates = []
        for i, (m, s) in enumerate(zip(masks, iou_scores)):
            m_bool = m.astype(bool)
            in_click = (0 <= py_i < m_bool.shape[0] and
                        0 <= px_i < m_bool.shape[1] and
                        m_bool[py_i, px_i])
            area = int(m_bool.sum())
            # 이미지 면적의 60% 초과하면 제외 (배경 방지)
            if in_click and area < img_area * 0.6:
                candidates.append((i, m_bool, float(s), area))

        if candidates:
            # 클릭 포함 + 적정 크기 → 가장 넓은 마스크 선택
            best_idx, best_mask_raw, best_score, _ = max(candidates, key=lambda t: t[3])
        else:
            # fallback: IOU 최고값
            best_idx = int(np.argmax(iou_scores))
            best_mask_raw = masks[best_idx].astype(bool)
            best_score = float(iou_scores[best_idx])

        best_mask = clean_mask(best_mask_raw)
        logger.info(
            f"IOU scores={[round(s,3) for s in iou_scores.tolist()]}, "
            f"선택 #{best_idx} score={best_score:.3f} area={best_mask.sum()}"
        )

        rows = np.where(np.any(best_mask, axis=1))[0]
        cols = np.where(np.any(best_mask, axis=0))[0]
        if len(rows) and len(cols):
            y1, y2, x1, x2 = int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1])
        else:
            y1, y2, x1, x2 = 0, orig_h, 0, orig_w

        detections = [{
            "id": 0,
            "score": round(best_score, 4),
            "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "mask_base64": mask_to_base64(best_mask),
        }]

    except Exception as e:
        logger.exception("포인트 세그멘테이션 오류")
        raise HTTPException(status_code=500, detail=f"추론 오류: {e}")

    return {
        "image_width": orig_w,
        "image_height": orig_h,
        "point": {"x": x, "y": y},
        "num_detections": 1,
        "detections": detections,
    }


@app.post("/segment-by-box")
async def segment_by_box(
    image: UploadFile = File(...),
    x1: float = Form(...),
    y1: float = Form(...),
    x2: float = Form(...),
    y2: float = Form(...),
):
    if interactive_predictor is None or processor is None:
        raise HTTPException(status_code=503, detail="모델 로딩 중입니다.")

    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 파일 오류: {e}")

    orig_w, orig_h = pil_image.size
    logger.info(f"박스 세그멘테이션: box=({x1},{y1},{x2},{y2})")

    try:
        state = processor.set_image(pil_image)
        _setup_interactive_predictor(state, orig_h, orig_w)

        box = np.array([x1, y1, x2, y2], dtype=np.float32)
        masks, iou_scores, _ = interactive_predictor.predict(
            box=box,
            multimask_output=False,
        )

        best_idx = int(np.argmax(iou_scores))
        best_mask = clean_mask(masks[best_idx].astype(bool))
        best_score = float(iou_scores[best_idx])
        logger.info(f"박스 세그멘테이션 score={best_score:.3f} area={best_mask.sum()}")

        rows = np.where(np.any(best_mask, axis=1))[0]
        cols = np.where(np.any(best_mask, axis=0))[0]
        if len(rows) and len(cols):
            ry1, ry2, rx1, rx2 = int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1])
        else:
            ry1, ry2, rx1, rx2 = 0, orig_h, 0, orig_w

        detections = [{
            "id": 0,
            "score": round(best_score, 4),
            "box": {"x1": rx1, "y1": ry1, "x2": rx2, "y2": ry2},
            "mask_base64": mask_to_base64(best_mask),
        }]

    except Exception as e:
        logger.exception("박스 세그멘테이션 오류")
        raise HTTPException(status_code=500, detail=f"추론 오류: {e}")

    return {
        "image_width": orig_w,
        "image_height": orig_h,
        "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "num_detections": 1,
        "detections": detections,
    }


@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    prompt: str = Form(default=""),
    confidence_threshold: float = Form(default=0.2),
    use_auto: bool = Form(default=False),
):
    if processor is None:
        raise HTTPException(status_code=503, detail="모델 로딩 중입니다.")

    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 파일 오류: {e}")

    orig_w, orig_h = pil_image.size
    prompt_tags = [t.strip() for t in prompt.split(",") if t.strip()]
    is_empty_prompt = not prompt_tags or prompt.strip() in ("", "object", "auto")
    logger.info(f"prompt='{prompt}' use_auto={use_auto} threshold={confidence_threshold}")

    try:
        base_state = processor.set_image(pil_image)
        all_detections = []

        if is_empty_prompt:
            # 프롬프트 없음: use_auto 여부에 따라 AUTO 전체 또는 빈 목록
            categories = AUTO_CATEGORIES if use_auto else []
            used_prompt = "auto" if use_auto else ""
            logger.info(f"프롬프트 없음 → {'자동 탐지 ' + str(len(categories)) + '개' if use_auto else '카테고리 없음'}")
        elif use_auto:
            # 프롬프트 있음 + AUTO 활성: 프롬프트 태그 + AUTO_CATEGORIES 합산 (중복 제거)
            seen: set[str] = set()
            categories = []
            for t in prompt_tags + AUTO_CATEGORIES:
                if t.lower() not in seen:
                    seen.add(t.lower())
                    categories.append(t)
            used_prompt = prompt.strip()
            logger.info(f"프롬프트 + AUTO: {len(prompt_tags)}개 태그 + AUTO {len(AUTO_CATEGORIES)}개 → {len(categories)}개 추론")
        else:
            # 프롬프트 있음 + AUTO 비활성: 프롬프트 태그만 사용
            categories = prompt_tags
            used_prompt = prompt.strip()
            logger.info(f"프롬프트만: {len(categories)}개 추론")

        for category in categories:
            dets = run_inference_with_state(base_state, category, confidence_threshold)
            all_detections.extend(dets)

        detections = nms_detections(all_detections, iou_threshold=0.2)
        logger.info(f"NMS 전 {len(all_detections)}개 → NMS 후 {len(detections)}개")

    except Exception as e:
        logger.exception("추론 오류")
        raise HTTPException(status_code=500, detail=f"추론 오류: {e}")

    for i, det in enumerate(detections):
        det["id"] = i

    return {
        "image_width": orig_w,
        "image_height": orig_h,
        "prompt": used_prompt,
        "auto_mode": use_auto,
        "num_detections": len(detections),
        "detections": detections,
    }
