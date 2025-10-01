
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from pathlib import Path
import json
import logging

# heavy libs (already in requirements)
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class EmbeddingConfig:
    # chunking-related
    heading_aware: bool = True
    cohesion_aware: bool = True
    cohesion_split: bool = True  # добавлено для совместимости
    chunk_target_tokens: int = 350
    chunk_max_tokens: int = 512
    chunk_min_tokens: int = 64
    min_chunk_tokens: int = 64  # алиас для совместимости
    overlap_sentences: int = 1
    sentence_overlap: int = 1  # алиас для совместимости
    table_as_is: bool = True

    # embeddings-related
    local_sbert_path: Optional[str] = None
    device: str = "cpu"
    batch_size: int = 64
    normalize: bool = True
    local_files_only: bool = True
    target_dimension: int = 1024  # целевая размерность эмбеддингов


class EmbeddingBackend:
    def __init__(self, cfg: EmbeddingConfig):
        self.cfg = cfg
        self._model: Optional[SentenceTransformer] = None
        self._tfidf: Optional[TfidfVectorizer] = None
        self._dim = None
        self._load_model()

    @property
    def model(self) -> Optional[SentenceTransformer]:
        return self._model

    @property
    def dimension(self) -> int:
        return int(self._dim)

    def _load_model(self):
        path = self.cfg.local_sbert_path
        if path:
            path_obj = Path(path)
            if not path_obj.exists():
                raise FileNotFoundError(
                    f"Указанный путь к модели SBERT не существует: '{path}'. "
                    "Проверьте настройку local_sbert_path."
                )

            try:
                model_kwargs = {"local_files_only": self.cfg.local_files_only}
                tokenizer_kwargs = {"local_files_only": self.cfg.local_files_only}
                self._model = SentenceTransformer(
                    str(path_obj),
                    device=self.cfg.device,
                    model_kwargs=model_kwargs,
                    tokenizer_kwargs=tokenizer_kwargs,
                )
            except OSError as exc:
                raise FileNotFoundError(
                    "Не удалось загрузить модель SBERT из каталога "
                    f"'{path}'. Убедитесь, что в директории присутствуют файлы "
                    "pytorch_model.bin или model.safetensors, а также config.json."
                ) from exc

            # infer dimension by encoding a dummy
            vec = self._model.encode(["test"])
            self._dim = int(vec.shape[1])
        else:
            # fallback: TF-IDF
            logging.warning(
                "Путь к локальной модели SBERT не указан. Используется запасной вариант TF-IDF."
            )
            self._tfidf = TfidfVectorizer(max_features=4096)
            self._dim = 4096

    def _resize_embeddings(self, embeddings: np.ndarray, target_dim: int) -> np.ndarray:
        """
        Приведение эмбеддингов к целевой размерности.
        Если размерность меньше целевой - дополняем нулями.
        Если больше - используем PCA или простое усечение.
        """
        current_dim = embeddings.shape[1]

        if current_dim == target_dim:
            return embeddings

        elif current_dim < target_dim:
            # Дополняем нулями справа
            padding = np.zeros((embeddings.shape[0], target_dim - current_dim))
            return np.hstack([embeddings, padding])

        else:
            # Усекаем до нужной размерности (можно использовать PCA для лучшего сохранения информации)
            # Простое усечение для скорости
            return embeddings[:, :target_dim]

    def encode(self, texts: List[str]) -> np.ndarray:
        if self._model is not None:
            arr = self._model.encode(texts, batch_size=self.cfg.batch_size,
                                     convert_to_numpy=True, normalize_embeddings=self.cfg.normalize)
        else:
            # TF-IDF lazy fit
            if getattr(self._tfidf, "vocabulary_", None) is None:
                self._tfidf.fit(texts)
            arr = self._tfidf.transform(texts).toarray()
            # l2 normalize for cosine
            if self.cfg.normalize:
                norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
                arr = arr / norms

        # Приводим к целевой размерности если указана
        if self.cfg.target_dimension and self.cfg.target_dimension != arr.shape[1]:
            arr = self._resize_embeddings(arr, self.cfg.target_dimension)
            # Обновляем размерность
            self._dim = self.cfg.target_dimension

        return arr


class EmbeddingsStore:
    """
    Simple on-disk index under outputs/index:
      - templates.json
      - chunks_meta.jsonl
      - embeddings.npy
    """

    def __init__(self, base_dir: str):
        self.base = Path(base_dir)
        (self.base/"index").mkdir(parents=True, exist_ok=True)

    def save_index(self, templates_meta: List[Dict[str, Any]], chunks_meta: List[Dict[str, Any]], embeddings: np.ndarray):
        (self.base/"index").mkdir(parents=True, exist_ok=True)
        (self.base/"index"/"templates.json").write_text(json.dumps(templates_meta,
                                                                   ensure_ascii=False, indent=2), encoding="utf-8")
        with open(self.base/"index"/"chunks_meta.jsonl", "w", encoding="utf-8") as f:
            for row in chunks_meta:
                f.write(json.dumps(row, ensure_ascii=False)+"\n")
        npy_path = self.base/"index"/"embeddings.npy"
        np.save(npy_path, embeddings)

    def load_index(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], np.ndarray]:
        templates = json.loads(
            (self.base/"index"/"templates.json").read_text(encoding="utf-8"))
        chunks_meta = []
        with open(self.base/"index"/"chunks_meta.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks_meta.append(json.loads(line))
        embs = np.load(self.base/"index"/"embeddings.npy")
        return templates, chunks_meta, embs
