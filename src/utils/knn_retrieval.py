"""
knn_retrieval.py

k-NN retrieval para few-shot NER y clasificación multilabel.
Usa FAISS + sentence-transformers para encontrar los ejemplos de entrenamiento
más similares a una query y usarlos como contexto en el prompt.

Basado en: GPT-NER (https://arxiv.org/pdf/2304.10428.pdf)

Uso típico para NER:
    retriever = KNNRetriever()
    retriever.build_index(train_tokens, train_tags)
    retriever.save_index("data/anat_em/knn_index")

    # En inference:
    retriever = KNNRetriever()
    retriever.load_index("data/anat_em/knn_index")
    examples = retriever.retrieve_examples("el ventrículo izquierdo", k=5)
    # examples: [{"sentence": ..., "entities": [...], "score": ...}, ...]
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm


class KNNRetriever:
    """
    Retriever de ejemplos similares usando embeddings de oraciones y FAISS.

    Compatible con:
      - NER: indexa oraciones con sus entidades BIO
      - Multilabel: indexa textos con sus etiquetas MeSH
    """

    def __init__(
        self,
        index_dir: Optional[str] = None,
        device: Optional[str] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.embedding_model = embedding_model
        self.index = None
        self.train_sentences: List[str] = []
        self.train_metadata: List = []   # entidades BIO o labels MeSH
        self.embeddings: Optional[np.ndarray] = None

        # Selección de dispositivo
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if index_dir and os.path.exists(index_dir):
            self.load_index(index_dir)

    # ------------------------------------------------------------------
    # Construcción del índice
    # ------------------------------------------------------------------

    def build_index(
        self,
        sentences: List[str],
        metadata: List,
        batch_size: int = 32,
    ) -> None:
        """
        Construye el índice FAISS a partir de una lista de oraciones.

        Args:
            sentences: Lista de strings a indexar.
            metadata: Lista paralela con metadatos por oración.
                      Para NER: list[list[tuple[str,str]]] (entidades)
                      Para multilabel: list[list[str]] (labels)
            batch_size: Tamaño de batch para encoding.
        """
        import faiss
        from sentence_transformers import SentenceTransformer

        print(f"Construyendo índice k-NN con {len(sentences)} ejemplos...")
        self.train_sentences = sentences
        self.train_metadata = metadata

        model = SentenceTransformer(self.embedding_model, device=self.device)
        self.embeddings = model.encode(
            sentences,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            device=self.device,
        ).astype(np.float32)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings)
        print(f"Índice construido con {self.index.ntotal} ejemplos.")

    def build_ner_index(
        self,
        train_tokens: List[List[str]],
        train_tags: List[List[str]],
        batch_size: int = 32,
    ) -> None:
        """
        Atajo para construir índice NER desde tokens/tags BIO.

        Args:
            train_tokens: Lista de oraciones tokenizadas.
            train_tags: Lista de secuencias de BIO labels correspondientes.
        """
        sentences = [" ".join(tokens) for tokens in train_tokens]
        metadata = [
            self._extract_entities_from_bio(tokens, tags)
            for tokens, tags in zip(train_tokens, train_tags)
        ]
        self.build_index(sentences, metadata, batch_size=batch_size)

    # ------------------------------------------------------------------
    # Recuperación
    # ------------------------------------------------------------------

    def retrieve_examples(
        self, query: str, k: int = 5
    ) -> List[Dict]:
        """
        Recupera los k ejemplos más similares a la query.

        Args:
            query: Oración o texto de consulta.
            k: Número de ejemplos a recuperar.

        Returns:
            Lista de dicts con keys: sentence, metadata, score.
        """
        if self.index is None:
            raise ValueError("Índice no construido. Llama a build_index() primero.")

        from sentence_transformers import SentenceTransformer

        # Cachear el modelo de embeddings para no recargarlo en cada query
        if not hasattr(self, "_query_model") or self._query_model is None:
            self._query_model = SentenceTransformer(self.embedding_model, device="cpu")

        query_emb = self._query_model.encode(
            [query],
            normalize_embeddings=True,
            device="cpu",
        ).astype(np.float32)

        scores, indices = self.index.search(query_emb, k)
        return [
            {
                "sentence": self.train_sentences[idx],
                "entities": self.train_metadata[idx],  # alias para NER
                "metadata": self.train_metadata[idx],
                "score": float(score),
            }
            for score, idx in zip(scores[0], indices[0])
            if idx >= 0
        ]

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def save_index(self, save_dir: str) -> None:
        """Guarda el índice FAISS y los metadatos en save_dir."""
        import faiss

        os.makedirs(save_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(save_dir, "faiss.index"))
        np.save(os.path.join(save_dir, "embeddings.npy"), self.embeddings)
        with open(os.path.join(save_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "train_sentences": self.train_sentences,
                    "train_metadata": self.train_metadata,
                    "embedding_model": self.embedding_model,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"Índice guardado en: {save_dir}")

    def load_index(self, index_dir: str) -> None:
        """Carga un índice FAISS previamente guardado."""
        import faiss

        self.index = faiss.read_index(os.path.join(index_dir, "faiss.index"))
        self.embeddings = np.load(os.path.join(index_dir, "embeddings.npy"))
        with open(os.path.join(index_dir, "metadata.json"), encoding="utf-8") as f:
            meta = json.load(f)
        self.train_sentences = meta["train_sentences"]
        self.train_metadata = meta["train_metadata"]
        self.embedding_model = meta.get("embedding_model", self.embedding_model)
        print(f"Índice cargado de: {index_dir} ({self.index.ntotal} ejemplos)")

    # ------------------------------------------------------------------
    # Utilidades internas
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_entities_from_bio(
        tokens: List[str], tags: List[str]
    ) -> List[Tuple[str, str]]:
        """Extrae entidades (texto, tipo) desde una secuencia BIO."""
        entities = []
        current_tokens: List[str] = []
        current_type: Optional[str] = None

        for token, tag in zip(tokens, tags):
            if tag.startswith("B-"):
                if current_tokens:
                    entities.append((" ".join(current_tokens), current_type))
                current_tokens = [token]
                current_type = tag[2:]
            elif tag.startswith("I-") and current_tokens:
                current_tokens.append(token)
            else:
                if current_tokens:
                    entities.append((" ".join(current_tokens), current_type))
                current_tokens = []
                current_type = None

        if current_tokens:
            entities.append((" ".join(current_tokens), current_type))

        return entities
