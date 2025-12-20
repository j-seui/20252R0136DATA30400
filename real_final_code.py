import os
import re
import csv
import json
import time
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, deque

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

import requests


# =========================
# User Config
# =========================
API_KEY = "AQ.Ab8RN6JUb82VrSIvjvHxX1wie7AH4c-SzUSd6pGwiHcCbRUt3g"

GEMINI_MODEL = "gemini-2.0-flash-lite-001"

ROOT_DIR = "Amazon_products"
CLASSES_PATH = os.path.join(ROOT_DIR, "classes.txt")
HIER_PATH = os.path.join(ROOT_DIR, "class_hierarchy.txt")
KW_PATH = os.path.join(ROOT_DIR, "class_related_keywords.txt")
TRAIN_CORPUS_PATH = os.path.join(ROOT_DIR, "train", "train_corpus.txt")
TEST_CORPUS_PATH = os.path.join(ROOT_DIR, "test", "test_corpus.txt")

ARTIFACT_DIR = os.path.join(ROOT_DIR, "_artifacts_roberta")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

CLASS_DESC_JSON = os.path.join(ARTIFACT_DIR, "class_descriptions.json")
CLASS_EMB_NPY = os.path.join(ARTIFACT_DIR, "class_embeddings.npy")
REVIEW_EMB_NPY = os.path.join(ARTIFACT_DIR, "review_embeddings.npy")
REVIEW_META_JSONL = os.path.join(ARTIFACT_DIR, "review_meta.jsonl")

CORE_LEAF_NPY = os.path.join(ARTIFACT_DIR, "core_leaf.npy")
CORE_LEAF_SOURCE_NPY = os.path.join(ARTIFACT_DIR, "core_leaf_src.npy")
CORE_LEAF_META_JSON = os.path.join(ARTIFACT_DIR, "core_leaf_meta.json")

# (Self-training) artifacts
SELF_TRAIN_CORE_LEAF_NPY = os.path.join(ARTIFACT_DIR, "self_train_core_leaf.npy")
SELF_TRAIN_CONF_NPY = os.path.join(ARTIFACT_DIR, "self_train_conf.npy")
SELF_TRAIN_META_JSON = os.path.join(ARTIFACT_DIR, "self_train_meta.json")

SUBMISSION_PATH = "submission.csv"

# API usage guard
API_CALL_LIMIT = 1000
API_CALL_COUNT = 0

# Embedding model
EMB_MODEL_NAME = "princeton-nlp/sup-simcse-roberta-large"
MAX_SEQ_LEN = 256
EMB_BATCH_SIZE = 64

# LLM fallback batching
TOPK_CANDIDATES_FOR_LLM = 5
LLM_BATCH_SIZE = 10

# Threshold calibration
PERCENTILE_FOR_THRESHOLD = 15
THRESHOLD_FLOOR = 0.20

# Parent greedy selection
PARENT_MIN_SIM = 0.05
DROP_ABS_DELTA = 0.25      # 절대 감소
DROP_REL_RATIO = 0.60      # 상대 감소(60% 이상 감소면 급감으로 판단)

# Self-training hyperparams
SELF_TRAIN_NUM_ITERS = 2
SELF_TRAIN_EPOCHS_PER_ITER = 4
SELF_TRAIN_LR = 2e-3
SELF_TRAIN_BATCH_SIZE = 256
SELF_TRAIN_WEIGHT_DECAY = 0.01
SELF_TRAIN_CONF_THRESHOLD = 0.90     # pseudo label 확신도
SELF_TRAIN_MAX_NEW_PER_ITER = 20000  # 한 iter에서 새로 편입할 최대 개수
SELF_TRAIN_SEED_MIN_SIM = None       # None이면 core threshold(thr) 사용


# =========================
# Reproducibility
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_int_str(x: str) -> bool:
    try:
        int(x)
        return True
    except Exception:
        return False


def safe_json_load(path: str, default=None):
    if default is None:
        default = {}
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def safe_json_save(obj: Any, path: str):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


# =========================
# Data Loading
# =========================
def load_classes(path: str) -> Tuple[Dict[int, str], Dict[str, int]]:
    """
    classes.txt: "class_id<TAB>class_name" 또는 "class_id class_name"
    """
    id2name = {}
    name2id = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            ln = line.strip()
            if not ln:
                continue
            parts = re.split(r"[\t ]+", ln, maxsplit=1)
            if len(parts) < 2:
                continue
            cid, cname = parts[0], parts[1].strip()
            if not is_int_str(cid):
                continue
            cid = int(cid)
            id2name[cid] = cname
            name2id[cname] = cid
    return id2name, name2id


def load_keywords(path: str) -> Dict[int, List[str]]:
    """
    class_related_keywords.txt: "class_id<TAB>kw1,kw2,..."
    """
    cid2kws = defaultdict(list)
    if not os.path.exists(path):
        return dict(cid2kws)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            ln = line.strip()
            if not ln:
                continue
            parts = ln.split("\t")
            if len(parts) < 2:
                continue
            cid = parts[0]
            if not is_int_str(cid):
                continue
            cid = int(cid)
            kws = re.split(r"[,\|;]+", parts[1])
            kws = [k.strip() for k in kws if k.strip()]
            cid2kws[cid].extend(kws)
    return dict(cid2kws)


def load_corpus_any(path: str, split_name: str):
    """
    train_corpus.txt / test_corpus.txt:
      - 최소: id<TAB>text
      - 혹시 label 컬럼이 있더라도: 마지막 컬럼을 text로 사용
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) == 2:
                pid, text = parts
            elif len(parts) >= 3:
                pid = parts[0]
                text = parts[-1]
            else:
                continue
            rows.append({"split": split_name, "pid": pid, "text": text})
    return rows


# =========================
# DAG Hierarchy (다중 parent 허용)
# (변수명은 parent_of / children_of 그대로 사용하되,
#  parent_of[child] = set(parents) 로 의미만 DAG로 확장)
# =========================
def load_hierarchy(path: str, num_classes: Optional[int] = None):
    """
    class_hierarchy.txt에서 (parent, child) 페어들을 읽고,
    DAG로 로딩 (다중 parent 허용).
    - parent_of: child -> set(parents)
    - children_of: parent -> set(children)
    사이클이 있으면 오류.
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    pairs = []
    max_id = -1
    for line in lines:
        parts = re.split(r"[\t, ]+", line)
        if len(parts) < 2:
            continue
        a, b = parts[0], parts[1]
        if is_int_str(a) and is_int_str(b):
            p = int(a)
            c = int(b)
            pairs.append((p, c))
            max_id = max(max_id, p, c)

    if num_classes is None:
        num_nodes = max_id + 1
    else:
        num_nodes = num_classes

    parent_of = defaultdict(set)   # child -> {parents}
    children_of = defaultdict(set) # parent -> {children}

    for p, c in pairs:
        if p == c:
            continue
        parent_of[c].add(p)
        children_of[p].add(c)

    # Kahn's algorithm for cycle check (DAG 검증)
    indeg = [0] * num_nodes
    for node in range(num_nodes):
        indeg[node] = len(parent_of.get(node, set()))

    q = deque([i for i in range(num_nodes) if indeg[i] == 0])
    visited = 0
    while q:
        u = q.popleft()
        visited += 1
        for v in children_of.get(u, set()):
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    if visited != num_nodes:
        raise ValueError("Hierarchy has a cycle (not a DAG). Please check class_hierarchy.txt")

    return parent_of, children_of


def get_leaf_nodes(children_of: Dict[int, set], num_classes: int) -> List[int]:
    leaves = []
    for cid in range(num_classes):
        if len(children_of.get(cid, set())) == 0:
            leaves.append(cid)
    return leaves


# =========================
# Vertex AI 호출
# =========================
def vertex_generate_content(prompt: str, temperature: float = 0.2, max_output_tokens: int = 2048) -> str:
    """
    Express mode non-stream endpoint:
      POST https://aiplatform.googleapis.com/v1/publishers/google/models/{model}:generateContent?key=API_KEY
    """
    global API_CALL_COUNT
    if API_CALL_COUNT >= API_CALL_LIMIT:
        raise RuntimeError(f"API_CALL_LIMIT exceeded ({API_CALL_LIMIT}).")

    url = (
        f"https://aiplatform.googleapis.com/v1/publishers/google/models/{GEMINI_MODEL}:generateContent"
        f"?key={API_KEY}"
    )

    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": prompt}]}
        ],
        "generationConfig": {
            "temperature": float(temperature),
            "maxOutputTokens": int(max_output_tokens),
        }
    }

    last_err = None
    for attempt in range(5):
        try:
            API_CALL_COUNT += 1
            resp = requests.post(url, json=payload, timeout=120)
            if resp.status_code != 200:
                last_err = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
                time.sleep(1.5 * (attempt + 1))
                continue
            data = resp.json()
            return _extract_text_from_generatecontent(data)
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"vertex_generate_content failed: {last_err}")


def _extract_text_from_generatecontent(data: dict) -> str:
    """
    generateContent 응답에서 텍스트를 최대한 안정적으로 추출
    """
    try:
        cands = data.get("candidates", [])
        if not cands:
            return ""
        parts = cands[0].get("content", {}).get("parts", [])
        texts = []
        for p in parts:
            t = p.get("text")
            if isinstance(t, str):
                texts.append(t)
        return "\n".join(texts).strip()
    except Exception:
        return ""


# =========================
# 1) Class description generation (cached)
# =========================
def build_desc_prompt(cid: int, cname: str, kws: List[str]) -> str:
    kw_str = ", ".join(kws[:30]) if kws else ""
    return (
        "You are given a product review classification label.\n"
        "Write a short, concrete description of what kinds of products/reviews belong to this label.\n"
        "Keep it concise but specific.\n\n"
        f"Label ID: {cid}\n"
        f"Label Name: {cname}\n"
        f"Related keywords: {kw_str}\n\n"
        "Return ONLY the description (one paragraph)."
    )


def parse_desc_response(text: str) -> str:
    # 과도한 포맷 제거
    t = text.strip()
    t = re.sub(r"^\s*[-*]\s*", "", t)
    t = t.strip()
    return t


def generate_class_descriptions(id2name: Dict[int, str], cid2kws: Dict[int, List[str]]) -> Dict[int, str]:
    """
    CLASS_DESC_JSON을 캐시로 사용:
    - 이미 존재하는 cid는 재호출하지 않음
    """
    desc_map = safe_json_load(CLASS_DESC_JSON, default={})
    # json 키는 str일 수 있으니 int로 normalize
    norm_map = {}
    for k, v in desc_map.items():
        if is_int_str(str(k)):
            norm_map[int(k)] = str(v)
    desc_map = norm_map

    missing = [cid for cid in id2name.keys() if cid not in desc_map or not str(desc_map[cid]).strip()]
    if not missing:
        return desc_map

    print(f"[Class desc] cached={len(desc_map)}, missing={len(missing)} (will call LLM)")
    for cid in tqdm(missing, desc="Generate class descriptions"):
        cname = id2name[cid]
        kws = cid2kws.get(cid, [])
        prompt = build_desc_prompt(cid, cname, kws)
        out = vertex_generate_content(prompt, temperature=0.2, max_output_tokens=512)
        desc = parse_desc_response(out)
        if not desc:
            desc = cname
        desc_map[cid] = desc
        # class 단위로 저장(중단/재개 가능)
        safe_json_save({str(k): v for k, v in desc_map.items()}, CLASS_DESC_JSON)

    return desc_map


# =========================
# 2) Embedding (cached)
# =========================
def mean_pool(last_hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    # last_hidden: [B, T, H], attn_mask: [B, T]
    mask = attn_mask.unsqueeze(-1).type_as(last_hidden)  # [B, T, 1]
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


def embed_texts_roberta(texts: List[str], device: str = "cuda" if torch.cuda.is_available() else "cpu") -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(EMB_MODEL_NAME)
    model = AutoModel.from_pretrained(EMB_MODEL_NAME).to(device)
    model.eval()

    outs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), EMB_BATCH_SIZE), desc=f"Embedding ({EMB_MODEL_NAME})"):
            batch = texts[i:i + EMB_BATCH_SIZE]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_SEQ_LEN,
                return_tensors="pt",
            ).to(device)
            out = model(**enc)
            pooled = mean_pool(out.last_hidden_state, enc["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            outs.append(pooled.detach().cpu().numpy())
    return np.vstack(outs).astype(np.float32)


def build_or_load_review_store(all_rows: List[dict]):
    """
    REVIEW_META_JSONL: split/pid/text 저장 (중단/재개 및 디버깅용)
    """
    # 이미 있으면 재생성하지 않음(원문 변경 없다는 가정)
    if os.path.exists(REVIEW_META_JSONL):
        return

    with open(REVIEW_META_JSONL, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_or_load_embeddings(
    class_desc_map: Dict[int, str],
    all_rows: List[dict],
    num_classes: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    CLASS_EMB_NPY, REVIEW_EMB_NPY 캐시 활용
    """
    # 1) class embeddings
    if os.path.exists(CLASS_EMB_NPY):
        class_emb = np.load(CLASS_EMB_NPY)
    else:
        class_texts = []
        for cid in range(num_classes):
            class_texts.append(str(class_desc_map.get(cid, "")))
        class_emb = embed_texts_roberta(class_texts)
        np.save(CLASS_EMB_NPY, class_emb)

    # 2) review embeddings
    if os.path.exists(REVIEW_EMB_NPY):
        review_emb = np.load(REVIEW_EMB_NPY)
    else:
        review_texts = [r["text"] for r in all_rows]
        review_emb = embed_texts_roberta(review_texts)
        np.save(REVIEW_EMB_NPY, review_emb)

    # 정규화(안전)
    class_emb = normalize_rows(class_emb.astype(np.float32))
    review_emb = normalize_rows(review_emb.astype(np.float32))
    return class_emb, review_emb


# =========================
# 3) Core leaf inference (cosine + LLM fallback) with checkpoint
# =========================
def calibrate_threshold(leaf_ids: List[int], leaf_emb: np.ndarray, review_emb_all: np.ndarray) -> float:
    """
    leaf top-1 cosine maxsim 분포에서 percentile 기반 threshold 산출
    """
    leaf_t = leaf_emb.T
    maxs = []
    bs = 2048
    for i in range(0, review_emb_all.shape[0], bs):
        chunk = review_emb_all[i:i + bs]
        sims = chunk @ leaf_t
        maxs.append(sims.max(axis=1))
    maxs = np.concatenate(maxs, axis=0)
    thr = float(np.percentile(maxs, PERCENTILE_FOR_THRESHOLD))
    thr = max(thr, THRESHOLD_FLOOR)
    return float(thr)


def build_llm_batch_prompt(batch_items: List[Tuple[int, str, List[Tuple[int, str]]]]) -> str:
    """
    batch_items: [(review_idx, review_text, [(cid, cname), ...topk])]
    LLM에게 각 review_idx별로 cid 하나를 선택하게 함
    """
    lines = []
    lines.append("You are a classifier. For each item, choose the best label ID from the candidates.\n")
    lines.append("Return STRICT JSON only: a list of objects with keys {idx, cid}.\n")
    lines.append("Do not include any extra text.\n\n")

    for idx, text, cands in batch_items:
        cand_str = "; ".join([f"{cid}:{cname}" for cid, cname in cands])
        lines.append(f"idx={idx}\nreview={text}\ncandidates={cand_str}\n")

    lines.append("\nJSON format example: [{\"idx\": 12, \"cid\": 5}, ...]")
    return "\n".join(lines)


def parse_llm_choice_response(text: str) -> List[Tuple[int, int]]:
    """
    LLM 응답에서 (idx, cid) 리스트를 복구
    """
    t = text.strip()
    # JSON 블록만 남기기 시도
    m = re.search(r"\[.*\]", t, flags=re.S)
    if m:
        t = m.group(0)
    try:
        data = json.loads(t)
        out = []
        if isinstance(data, list):
            for it in data:
                if isinstance(it, dict) and "idx" in it and "cid" in it:
                    out.append((int(it["idx"]), int(it["cid"])))
        return out
    except Exception:
        return []


def infer_core_leaf_classes(
    leaf_ids, leaf_emb, review_emb_all, all_reviews, id2name, class_desc_map
):
    """
    체크포인트:
      - CORE_LEAF_NPY: 리뷰 idx -> core leaf class_id
      - CORE_LEAF_SOURCE_NPY: 0=cosine, 1=llm
      - CORE_LEAF_META_JSON: threshold 및 leaf_ids 검증
    중단되더라도 저장되므로 재실행 시 이어서 진행.
    """
    global API_CALL_COUNT

    leaf_ids = list(leaf_ids)
    leaf_t = leaf_emb.T
    n = review_emb_all.shape[0]

    # 1) threshold
    thr = calibrate_threshold(leaf_ids, leaf_emb, review_emb_all)

    # 2) load or init checkpoint
    ok = False
    if os.path.exists(CORE_LEAF_NPY) and os.path.exists(CORE_LEAF_SOURCE_NPY) and os.path.exists(CORE_LEAF_META_JSON):
        meta = safe_json_load(CORE_LEAF_META_JSON, default={})
        if int(meta.get("leaf_count", -1)) == len(leaf_ids) and float(meta.get("threshold", -1.0)) == float(thr):
            ok = True

    if ok:
        core_leaf = np.load(CORE_LEAF_NPY).astype(np.int32)
        core_src = np.load(CORE_LEAF_SOURCE_NPY).astype(np.uint8)
        if len(core_leaf) != n:
            ok = False

    if not ok:
        core_leaf = np.empty(n, dtype=np.int32)
        core_src = np.zeros(n, dtype=np.uint8)

        bs = 2048
        for i in tqdm(range(0, n, bs), desc="(Checkpoint) cosine-top1 core leaf"):
            chunk = review_emb_all[i:i + bs]
            sims = chunk @ leaf_t
            arg = sims.argmax(axis=1)
            core_leaf[i:i + bs] = np.array([leaf_ids[int(p)] for p in arg], dtype=np.int32)

        np.save(CORE_LEAF_NPY, core_leaf)
        np.save(CORE_LEAF_SOURCE_NPY, core_src)
        safe_json_save({"threshold": thr, "leaf_count": len(leaf_ids)}, CORE_LEAF_META_JSON)

    # 3) LLM 보정: maxsim < thr 인 리뷰만
    ambiguous_idx = []
    bs = 2048
    for i in range(0, n, bs):
        chunk = review_emb_all[i:i + bs]
        sims = chunk @ leaf_t
        maxsim = sims.max(axis=1)
        for j, s in enumerate(maxsim):
            idx = i + j
            if float(s) < float(thr) and core_src[idx] != 1:
                ambiguous_idx.append(idx)

    print(f"[Ambiguous remaining] {len(ambiguous_idx)} items require LLM (batched {LLM_BATCH_SIZE}).")
    if not ambiguous_idx:
        return {i: int(core_leaf[i]) for i in range(n)}

    # LLM 배치 처리(중단/재개: core_src==1은 skip)
    for st in tqdm(range(0, len(ambiguous_idx), LLM_BATCH_SIZE), desc="LLM disambiguation"):
        batch = ambiguous_idx[st:st + LLM_BATCH_SIZE]
        batch_items = []
        for idx in batch:
            text = all_reviews[idx]["text"]
            # topk candidates by cosine
            sims = review_emb_all[idx] @ leaf_t
            topk = np.argsort(-sims)[:TOPK_CANDIDATES_FOR_LLM]
            cands = [(leaf_ids[int(j)], id2name.get(int(leaf_ids[int(j)]), str(leaf_ids[int(j)]))) for j in topk]
            batch_items.append((idx, text, cands))

        prompt = build_llm_batch_prompt(batch_items)
        resp = vertex_generate_content(prompt, temperature=0.0, max_output_tokens=1024)
        choices = parse_llm_choice_response(resp)

        # 적용
        for i2, cid in choices:
            i2 = int(i2)
            if 0 <= i2 < n:
                core_leaf[i2] = int(cid)
                core_src[i2] = 1

        # checkpoint
        np.save(CORE_LEAF_NPY, core_leaf)
        np.save(CORE_LEAF_SOURCE_NPY, core_src)

    return {i: int(core_leaf[i]) for i in range(n)}


# =========================
# 3.5) Self-training to refine core leaf
# =========================
class PseudoLeafDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class LinearLeafClassifier(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def _compute_leaf_maxsim(review_emb_all: np.ndarray, leaf_emb: np.ndarray) -> np.ndarray:
    """
    review별 leaf에 대한 max cosine(sim) (배치로 계산)
    """
    leaf_t = leaf_emb.T
    n = review_emb_all.shape[0]
    out = np.empty(n, dtype=np.float32)
    bs = 2048
    for i in range(0, n, bs):
        chunk = review_emb_all[i:i + bs]
        sims = chunk @ leaf_t
        out[i:i + bs] = sims.max(axis=1).astype(np.float32)
    return out


def self_train_refine_core_leaf(
    leaf_ids: List[int],
    leaf_emb: np.ndarray,
    review_emb_all: np.ndarray,
    core_leaf_map: Dict[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    초기 core_leaf(cosine/LLM)를 seed로 삼아 leaf-classifier를 학습하고,
    높은 confidence 예측을 pseudo label로 편입하는 self-training 수행.

    반환:
      - st_core_leaf: (N,) int32 (refine된 core leaf class_id)
      - st_conf: (N,) float32 (해당 예측 confidence)
    """
    n = review_emb_all.shape[0]
    leaf_ids = list(leaf_ids)
    leaf_id_to_index = {cid: i for i, cid in enumerate(leaf_ids)}

    # 코어/소스 로드(기존 캐시 활용)
    if os.path.exists(CORE_LEAF_NPY) and os.path.exists(CORE_LEAF_SOURCE_NPY) and os.path.exists(CORE_LEAF_META_JSON):
        core_leaf = np.load(CORE_LEAF_NPY).astype(np.int32)
        core_src = np.load(CORE_LEAF_SOURCE_NPY).astype(np.uint8)
        meta = safe_json_load(CORE_LEAF_META_JSON, default={})
        thr = float(meta.get("threshold", THRESHOLD_FLOOR))
    else:
        # 비정상 케이스: map로 대체
        core_leaf = np.array([core_leaf_map[i] for i in range(n)], dtype=np.int32)
        core_src = np.zeros(n, dtype=np.uint8)
        thr = THRESHOLD_FLOOR

    # self-training 재개 체크
    if os.path.exists(SELF_TRAIN_CORE_LEAF_NPY) and os.path.exists(SELF_TRAIN_CONF_NPY) and os.path.exists(SELF_TRAIN_META_JSON):
        st_meta = safe_json_load(SELF_TRAIN_META_JSON, default={})
        st_core_leaf = np.load(SELF_TRAIN_CORE_LEAF_NPY).astype(np.int32)
        st_conf = np.load(SELF_TRAIN_CONF_NPY).astype(np.float32)
        if len(st_core_leaf) == n and len(st_conf) == n and int(st_meta.get("done", 0)) == 1:
            print("[Self-training] Found completed cache. Skip training.")
            return st_core_leaf, st_conf

    # seed 선정: (LLM 확정) OR (cosine maxsim >= thr)
    leaf_maxsim = _compute_leaf_maxsim(review_emb_all, leaf_emb)
    seed_thr = thr if SELF_TRAIN_SEED_MIN_SIM is None else float(SELF_TRAIN_SEED_MIN_SIM)
    seed_mask = (core_src == 1) | (leaf_maxsim >= float(seed_thr))

    seed_idx = np.where(seed_mask)[0]
    if seed_idx.size == 0:
        # fallback
        seed_idx = np.arange(n)

    # 현재 레이블 -> leaf index로 변환 (seed만)
    seed_y = []
    seed_x = []
    for i in seed_idx.tolist():
        cid = int(core_leaf[i])
        if cid in leaf_id_to_index:
            seed_x.append(review_emb_all[i])
            seed_y.append(leaf_id_to_index[cid])
    if len(seed_x) < 100:
        print("[Self-training] Too few seed labels. Skip self-training and use base core_leaf.")
        st_core_leaf = core_leaf.copy().astype(np.int32)
        st_conf = np.ones(n, dtype=np.float32) * 0.0
        # base confidence: leaf_maxsim (정규화 아님)
        st_conf = np.clip(leaf_maxsim, 0.0, 1.0).astype(np.float32)
        np.save(SELF_TRAIN_CORE_LEAF_NPY, st_core_leaf)
        np.save(SELF_TRAIN_CONF_NPY, st_conf)
        safe_json_save({"done": 1, "reason": "too_few_seed"}, SELF_TRAIN_META_JSON)
        return st_core_leaf, st_conf

    seed_x = np.stack(seed_x, axis=0).astype(np.float32)
    seed_y = np.array(seed_y, dtype=np.int64)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_dim = review_emb_all.shape[1]
    num_leaf = len(leaf_ids)

    model = LinearLeafClassifier(in_dim=in_dim, num_classes=num_leaf).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=SELF_TRAIN_LR, weight_decay=SELF_TRAIN_WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss()

    # self-training state
    st_core_leaf = core_leaf.copy().astype(np.int32)
    st_conf = np.zeros(n, dtype=np.float32)

    for it in range(SELF_TRAIN_NUM_ITERS):
        # (iter별) 학습 데이터 구성: 현재 st_core_leaf 기반에서 confidence 높은 것만
        # 초기 iter에서는 seed만. 이후에는 st_conf가 충분히 높은 것들까지 포함.
        if it == 0:
            train_mask = seed_mask
        else:
            train_mask = seed_mask | (st_conf >= float(SELF_TRAIN_CONF_THRESHOLD))

        train_idx = np.where(train_mask)[0]
        train_x = []
        train_y = []
        for i in train_idx.tolist():
            cid = int(st_core_leaf[i])
            if cid in leaf_id_to_index:
                train_x.append(review_emb_all[i])
                train_y.append(leaf_id_to_index[cid])

        if len(train_x) < 100:
            print(f"[Self-training] Iter {it}: too few training samples ({len(train_x)}). Stop.")
            break

        train_x = np.stack(train_x, axis=0).astype(np.float32)
        train_y = np.array(train_y, dtype=np.int64)

        ds = PseudoLeafDataset(train_x, train_y)
        dl = DataLoader(ds, batch_size=SELF_TRAIN_BATCH_SIZE, shuffle=True, drop_last=False)

        model.train()
        for ep in range(SELF_TRAIN_EPOCHS_PER_ITER):
            total_loss = 0.0
            for xb, yb in dl:
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = crit(logits, yb)
                loss.backward()
                opt.step()
                total_loss += float(loss.item()) * xb.size(0)
            avg_loss = total_loss / max(len(ds), 1)
            print(f"[Self-training] iter={it} epoch={ep} loss={avg_loss:.4f} train_n={len(ds)}")

        # 예측 및 pseudo label 편입
        model.eval()
        new_candidates = []

        bs = 4096
        with torch.no_grad():
            for i in range(0, n, bs):
                xb = torch.from_numpy(review_emb_all[i:i + bs]).float().to(device)
                logits = model(xb)
                probs = torch.softmax(logits, dim=1)
                conf, pred = torch.max(probs, dim=1)
                conf = conf.detach().cpu().numpy().astype(np.float32)
                pred = pred.detach().cpu().numpy().astype(np.int32)
                for j in range(len(conf)):
                    idx = i + j
                    c = float(conf[j])
                    if c >= float(SELF_TRAIN_CONF_THRESHOLD):
                        new_cid = int(leaf_ids[int(pred[j])])
                        new_candidates.append((c, idx, new_cid))

        # 상위 confidence부터 최대 N개 편입
        new_candidates.sort(key=lambda x: -x[0])
        added = 0
        for c, idx, new_cid in new_candidates:
            # seed는 그대로 두되, seed가 아닌 경우에 주로 업데이트
            if seed_mask[idx]:
                # seed와 동일 라벨이면 conf만 갱신
                if int(st_core_leaf[idx]) == int(new_cid):
                    st_conf[idx] = max(st_conf[idx], float(c))
                continue

            # 업데이트
            st_core_leaf[idx] = int(new_cid)
            st_conf[idx] = max(st_conf[idx], float(c))
            added += 1
            if added >= int(SELF_TRAIN_MAX_NEW_PER_ITER):
                break

        print(f"[Self-training] iter={it}: added_new={added}, total_highconf={(st_conf>=SELF_TRAIN_CONF_THRESHOLD).sum()}")

        # iter checkpoint
        np.save(SELF_TRAIN_CORE_LEAF_NPY, st_core_leaf)
        np.save(SELF_TRAIN_CONF_NPY, st_conf)
        safe_json_save({"done": 0, "iter": it, "added": added}, SELF_TRAIN_META_JSON)

    # 완료 체크포인트
    np.save(SELF_TRAIN_CORE_LEAF_NPY, st_core_leaf)
    np.save(SELF_TRAIN_CONF_NPY, st_conf)
    safe_json_save({"done": 1}, SELF_TRAIN_META_JSON)
    return st_core_leaf, st_conf


# =========================
# 4) Greedy parent selection (max 3 labels) - DAG 대응
# =========================
def select_hierarchical_labels(
    core_leaf_map: Dict[int, int],
    review_emb_all: np.ndarray,
    class_emb_all: np.ndarray,
    parent_of: Dict[int, set],
    all_rows: List[dict],
    max_labels: int = 3
) -> Dict[str, List[int]]:
    """
    DAG에서 parent_of[child]가 set(parents) 이므로,
    각 단계에서 '가장 유사한 parent'를 선택하는 방식으로 chain을 구성.
    (leaf, best-parent, best-grandparent) 최대 3개.
    """
    pid_to_labels = {}
    for idx, r in enumerate(tqdm(all_rows, desc="Select hierarchical labels (DAG)")):
        pid = r["pid"]
        core = int(core_leaf_map[idx])
        labels = [core]

        cur = core
        prev_sim = float(review_emb_all[idx] @ class_emb_all[cur])

        for _ in range(max_labels - 1):
            parents = list(parent_of.get(cur, set()))
            if not parents:
                break

            # parent 후보 중 best
            best_p = None
            best_sim = -1.0
            for p in parents:
                s = float(review_emb_all[idx] @ class_emb_all[int(p)])
                if s > best_sim:
                    best_sim = s
                    best_p = int(p)

            if best_p is None:
                break

            # 중단 규칙
            if best_sim < float(PARENT_MIN_SIM):
                break
            abs_drop = float(prev_sim - best_sim)
            rel_drop = abs_drop / max(float(prev_sim), 1e-9)
            if abs_drop >= float(DROP_ABS_DELTA):
                break
            if rel_drop >= float(DROP_REL_RATIO):
                break

            labels.append(int(best_p))
            cur = int(best_p)
            prev_sim = float(best_sim)

        pid_to_labels[pid] = labels
    return pid_to_labels


def write_submission(test_rows, all_rows, review_emb_all, leaf_ids, leaf_emb, pid_to_labels, out_path):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "labels"])
        for r in test_rows:
            pid = r["pid"]
            labels = pid_to_labels.get(pid, [])
            if not labels:
                # fallback: 최소 1개 라벨
                idx = None
                for i, rr in enumerate(all_rows):
                    if rr.get("split") == "test" and rr.get("pid") == pid:
                        idx = i
                        break
                if idx is not None:
                    sims = review_emb_all[idx] @ leaf_emb.T
                    cid = leaf_ids[int(np.argmax(sims))]
                    labels = [int(cid)]
            w.writerow([pid, ",".join(map(str, labels))])


# =========================
# Main
# =========================
def main():
    set_seed(42)

    # 0) Load data
    id2name, _ = load_classes(CLASSES_PATH)
    num_classes = len(id2name)
    cid2kws = load_keywords(KW_PATH)

    # 1) DAG hierarchy (다중 parent)
    parent_of, children_of = load_hierarchy(HIER_PATH, num_classes=num_classes)
    leaf_ids = get_leaf_nodes(children_of, num_classes=num_classes)

    print(f"[Hierarchy] num_classes={num_classes}, leaves={len(leaf_ids)}")

    # 2) Class descriptions (cached)
    class_desc_map = generate_class_descriptions(id2name, cid2kws)

    # 3) Corpus
    train_rows = load_corpus_any(TRAIN_CORPUS_PATH, "train")
    test_rows = load_corpus_any(TEST_CORPUS_PATH, "test")
    all_rows = train_rows + test_rows
    build_or_load_review_store(all_rows)

    # 4) Embeddings (cached)
    class_emb_all, review_emb_all = build_or_load_embeddings(class_desc_map, all_rows, num_classes)

    # 5) Leaf embeddings
    leaf_emb = class_emb_all[np.array(leaf_ids, dtype=np.int32)]

    # 6) Core leaf inference (cosine + LLM fallback, checkpoint)
    core_leaf_map = infer_core_leaf_classes(
        leaf_ids=leaf_ids,
        leaf_emb=leaf_emb,
        review_emb_all=review_emb_all,
        all_reviews=all_rows,
        id2name=id2name,
        class_desc_map=class_desc_map,
    )

    # 7) Self-training refine (checkpoint)
    st_core_leaf, st_conf = self_train_refine_core_leaf(
        leaf_ids=leaf_ids,
        leaf_emb=leaf_emb,
        review_emb_all=review_emb_all,
        core_leaf_map=core_leaf_map,
    )
    # 최종 core_leaf_map 갱신(인덱스 -> class_id)
    core_leaf_map = {i: int(st_core_leaf[i]) for i in range(len(st_core_leaf))}

    # 8) Hierarchical label selection (DAG)
    pid_to_labels = select_hierarchical_labels(
        core_leaf_map=core_leaf_map,
        review_emb_all=review_emb_all,
        class_emb_all=class_emb_all,
        parent_of=parent_of,
        all_rows=all_rows,
        max_labels=3,
    )

    # 9) Write submission
    write_submission(
        test_rows=test_rows,
        all_rows=all_rows,
        review_emb_all=review_emb_all,
        leaf_ids=leaf_ids,
        leaf_emb=leaf_emb,
        pid_to_labels=pid_to_labels,
        out_path=SUBMISSION_PATH,
    )

    print(f"[Done] wrote {SUBMISSION_PATH}")
    print(f"[API calls] {API_CALL_COUNT}/{API_CALL_LIMIT}")


if __name__ == "__main__":
    main()
