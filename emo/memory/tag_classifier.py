"""TagClassifier — 标签分类器（基于 label1/label2 体系）

做梦时:
  1. 从已结构化的记忆中收集 (user_message → label1, label2) 训练对
  2. 用 sentence-transformers 编码 user_message
  3. 训练 sklearn LogisticRegression 分类器

在线推理时:
  用户输入 → 预测 label1 → 预测 label2 → 缩小记忆检索范围

label1（大类）: conversation, knowledge, opinion, fact, capability, ...
label2（次级类）: health, sports, food, relationship, daily_life, ...
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class TagClassifier:
    """标签分类器：基于 label1/label2 体系"""

    def __init__(
        self,
        persist_store,
        llm_client=None,
        llm_model: str = "qwen-plus",
        model_dir: str = "emo/memory/models",
    ):
        self._persist = persist_store
        self._llm = llm_client
        self._model_name = llm_model
        self._model_dir = Path(model_dir)
        self._model_dir.mkdir(parents=True, exist_ok=True)

        # 运行时状态
        self._embed_model = None
        self._clf_label1 = None  # sklearn classifier for label1
        self._clf_label2 = None  # sklearn classifier for label2
        self._label1_list: List[str] = []
        self._label2_list: List[str] = []

        # 加载已有模型
        self._load_model()

    # ── 训练 ──

    def train(self) -> bool:
        """从已结构化的记忆中训练 label1 和 label2 分类器

        Returns:
            是否训练成功
        """
        # 加载 embedding 模型
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer
            model_path = "emo/models/all-MiniLM-L6-v2"
            if os.path.isdir(model_path):
                self._embed_model = SentenceTransformer(model_path)
            else:
                self._embed_model = SentenceTransformer("all-MiniLM-L6-v2")

        # 收集训练数据
        X_texts = []
        y_label1 = []
        y_label2 = []

        memories = (
            self._persist.get_by_status("dreamed") +
            self._persist.get_by_status("settled")
        )

        for mf in memories:
            if not mf.label1 or not mf.label2 or not mf.raw_content:
                continue

            user_msg = self._extract_user_message(mf.raw_content)
            if not user_msg:
                continue

            X_texts.append(user_msg)
            y_label1.append(mf.label1)
            y_label2.append(mf.label2)

        if len(X_texts) < 20:
            logger.warning(f"Not enough training data ({len(X_texts)} samples)")
            return False

        logger.info(
            f"Training on {len(X_texts)} samples: "
            f"{len(set(y_label1))} label1 classes, "
            f"{len(set(y_label2))} label2 classes"
        )

        # 编码
        X_embeds = self._embed_model.encode(X_texts, normalize_embeddings=True)

        # 训练 label1 分类器
        self._label1_list = sorted(set(y_label1))
        label1_to_idx = {l: i for i, l in enumerate(self._label1_list)}
        y1_idx = np.array([label1_to_idx[l] for l in y_label1])
        self._clf_label1 = self._train_classifier(X_embeds, y1_idx)

        # 训练 label2 分类器
        self._label2_list = sorted(set(y_label2))
        label2_to_idx = {l: i for i, l in enumerate(self._label2_list)}
        y2_idx = np.array([label2_to_idx[l] for l in y_label2])
        self._clf_label2 = self._train_classifier(X_embeds, y2_idx)

        # 保存
        self._save_model()

        # 打印统计
        print(f"  Label1 classes ({len(self._label1_list)}): {self._label1_list}")
        print(f"  Label2 classes ({len(self._label2_list)}): {self._label2_list[:15]}...")

        # 训练集准确率
        acc1 = np.mean(self._clf_label1.predict(X_embeds) == y1_idx)
        acc2 = np.mean(self._clf_label2.predict(X_embeds) == y2_idx)
        print(f"  Train accuracy: label1={acc1:.3f}, label2={acc2:.3f}")

        return True

    def _train_classifier(self, X: np.ndarray, y: np.ndarray):
        """训练 sklearn LogisticRegression"""
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=300, C=1.0, solver="lbfgs")
        clf.fit(X, y)
        return clf

    def _extract_user_message(self, raw_content: str) -> Optional[str]:
        """从 raw_content 中提取用户消息

        支持多种格式:
        1. [User] ... 或 User: ... (Aditi 格式)
        2. SPEAKER: ... + TEXT: ... (LoCoMo 格式)
        """
        lines = raw_content.split("\n")

        # 尝试 LoCoMo 格式: 找 TEXT: 行
        text_line = None
        for line in lines:
            line = line.strip()
            if line.startswith("TEXT:"):
                text_line = line.split(":", 1)[-1].strip()
                break

        if text_line:
            return text_line

        # 尝试 Aditi 格式: [User] 或 User:
        for line in lines:
            line = line.strip()
            if line.startswith("[User]") or line.startswith("User:"):
                msg = line.split("]", 1)[-1].strip() if "]" in line else line.split(":", 1)[-1].strip()
                if msg:
                    return msg

        return None

    # ── 推理 ──

    def predict(
        self, user_message: str, top_k: int = 2, confidence_threshold: float = 0.5
    ) -> dict:
        """预测用户消息的 label1 和 label2

        Args:
            user_message: 用户输入
            top_k: 返回前 K 个类别
            confidence_threshold: 置信度阈值，低于此值标记为 unknown

        Returns:
            dict:
              label1: List of (label_name, confidence_score)
              label2: List of (label_name, confidence_score)
              known: bool — 是否命中已知类别
              unknown_query: str — 如果 unknown，保存原始 query 供做梦拓展
        """
        if self._clf_label1 is None or self._clf_label2 is None:
            logger.warning("Classifier not loaded")
            return {"label1": [], "label2": [], "known": False, "unknown_query": user_message}

        # 延迟加载 embedding 模型
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer
            model_path = "emo/models/all-MiniLM-L6-v2"
            if os.path.isdir(model_path):
                self._embed_model = SentenceTransformer(model_path)
            else:
                self._embed_model = SentenceTransformer("all-MiniLM-L6-v2")

        embed = self._embed_model.encode([user_message], normalize_embeddings=True)

        # 预测 label1
        probs1 = self._clf_label1.predict_proba(embed)[0]
        top1_idx = np.argsort(probs1)[::-1][:top_k]
        label1_preds = [
            (self._label1_list[idx], float(probs1[idx]))
            for idx in top1_idx
        ]

        # 预测 label2
        probs2 = self._clf_label2.predict_proba(embed)[0]
        top2_idx = np.argsort(probs2)[::-1][:top_k]
        label2_preds = [
            (self._label2_list[idx], float(probs2[idx]))
            for idx in top2_idx
        ]

        # 判断是否命中已知类别（L1 和 L2 都要超过阈值）
        l1_conf = label1_preds[0][1] if label1_preds else 0.0
        l2_conf = label2_preds[0][1] if label2_preds else 0.0
        known = (l1_conf >= confidence_threshold) and (l2_conf >= confidence_threshold)

        return {
            "label1": label1_preds,
            "label2": label2_preds,
            "known": known,
            "unknown_query": user_message if not known else "",
        }

    # ── 保存/加载 ──

    @classmethod
    def load(cls, model_path: str) -> "TagClassifier":
        """从文件加载已训练好的分类器

        Args:
            model_path: 模型文件路径（.pkl）

        Returns:
            TagClassifier 实例
        """
        instance = cls.__new__(cls)
        instance._persist = None
        instance._llm = None
        instance._model_name = None
        instance._model_dir = Path(model_path).parent
        instance._embed_model = None
        instance._clf_label1 = None
        instance._clf_label2 = None
        instance._label1_list = []
        instance._label2_list = []

        # 加载模型
        with open(model_path, "rb") as f:
            config = pickle.load(f)
        instance._clf_label1 = config.get("clf_label1")
        instance._clf_label2 = config.get("clf_label2")
        instance._label1_list = config.get("label1_list", [])
        instance._label2_list = config.get("label2_list", [])

        if instance._clf_label1:
            logger.info(
                f"Loaded classifier: "
                f"{len(instance._label1_list)} label1, "
                f"{len(instance._label2_list)} label2"
            )

        return instance

    def _save_model(self) -> None:
        model_path = self._model_dir / "tag_classifier.pkl"
        config = {
            "clf_label1": self._clf_label1,
            "clf_label2": self._clf_label2,
            "label1_list": self._label1_list,
            "label2_list": self._label2_list,
        }
        with open(model_path, "wb") as f:
            pickle.dump(config, f)
        logger.info(f"Saved classifier to {model_path}")

    def _load_model(self) -> None:
        model_path = self._model_dir / "tag_classifier.pkl"
        if model_path.exists():
            with open(model_path, "rb") as f:
                config = pickle.load(f)
            self._clf_label1 = config.get("clf_label1")
            self._clf_label2 = config.get("clf_label2")
            self._label1_list = config.get("label1_list", [])
            self._label2_list = config.get("label2_list", [])
            if self._clf_label1:
                logger.info(
                    f"Loaded classifier: "
                    f"{len(self._label1_list)} label1, "
                    f"{len(self._label2_list)} label2"
                )

    def stats(self) -> dict:
        return {
            "label1_classes": len(self._label1_list),
            "label2_classes": len(self._label2_list),
            "label1_list": self._label1_list,
            "label2_list": self._label2_list[:15],
            "loaded": self._clf_label1 is not None,
        }
