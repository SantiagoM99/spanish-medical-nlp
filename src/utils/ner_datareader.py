"""
ner_datareader.py

Carga el dataset AnatEM Spanish NER desde data/anat_em/.

Estructura esperada en data/anat_em/:
    nersuite/                    <- archivos .nersuite (uno por documento)
        PMID-10080451.nersuite
        PMC-1072800-sec-07.nersuite
        ...
    train-documents.list         <- IDs de documentos del split train
    devel-documents.list         <- IDs de documentos del split dev
    test-documents.list          <- IDs de documentos del split test

Formato .nersuite (dos columnas, separadas por tab, línea vacía = nueva oración):
    fibrilación\tO
    ventricular\tB-Multi-tissue_structure
    ...

Uso:
    dataset = NERDataset("data/anat_em")
    print(dataset)
    tokens, labels = dataset.get_tokens_and_labels("train")
"""

from pathlib import Path


class NERDataset:
    """
    Carga y sirve el dataset AnatEM Spanish NER.

    Attributes:
        data_dir  : Path al directorio raíz (data/anat_em/)
        label2id  : dict label → int
        id2label  : dict int → label
        train / dev / test : list[list[tuple[str, str]]]
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self._validate()

        nersuite_dir = self.data_dir / "nersuite"

        self.train = self._load_split(nersuite_dir, self.data_dir / "train-documents.list")
        self.dev   = self._load_split(nersuite_dir, self.data_dir / "devel-documents.list")
        self.test  = self._load_split(nersuite_dir, self.data_dir / "test-documents.list")

        self.label2id, self.id2label = self._build_vocab()

    def _validate(self) -> None:
        nersuite_dir = self.data_dir / "nersuite"
        if not nersuite_dir.is_dir():
            raise FileNotFoundError(
                f"No se encontró {nersuite_dir}.\n"
                "Copia los archivos .nersuite a data/anat_em/nersuite/"
            )
        for fname in ["train-documents.list", "devel-documents.list", "test-documents.list"]:
            if not (self.data_dir / fname).exists():
                raise FileNotFoundError(
                    f"{fname} no encontrado en {self.data_dir}.\n"
                    "Copia los .list de AnatEM/splits/ a data/anat_em/"
                )

    def _load_split(self, nersuite_dir: Path, list_file: Path) -> list:
        """Carga los documentos de un split según su lista."""
        doc_ids = [
            line.strip()
            for line in list_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        sentences = []
        missing = []
        for doc_id in doc_ids:
            filepath = nersuite_dir / f"{doc_id}.nersuite"
            if not filepath.exists():
                missing.append(doc_id)
                continue
            sentences.extend(self._parse_nersuite(filepath))

        if missing:
            print(f"  AVISO: {len(missing)} archivos no encontrados (ej: {missing[:3]})")
        return sentences

    @staticmethod
    def _clean_label(label: str) -> str:
        label = label.strip()
        if label in ("", "=", "o", "O-e", "o-e"):
            return "O"
        label = label.replace("Multi_tissue_structure", "Multi-tissue_structure")
        return label

    def _parse_nersuite(self, filepath: Path) -> list:
        """
        Parsea un archivo .nersuite a lista de oraciones.

        Returns:
            list[list[tuple[str, str]]] — oraciones de (token, label)
        """
        sentences = []
        current = []
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line == "":
                    if current:
                        sentences.append(current)
                        current = []
                else:
                    parts = line.split("\t")
                    token = parts[0]
                    label = self._clean_label(parts[1]) if len(parts) > 1 else "O"
                    current.append((token, label))
        if current:
            sentences.append(current)
        return sentences

    def _build_vocab(self) -> tuple:
        labels = set()
        for split in [self.train, self.dev, self.test]:
            for sent in split:
                for _, lbl in sent:
                    labels.add(lbl)
        sorted_labels = ["O"] + sorted(lbl for lbl in labels if lbl != "O")
        label2id = {lbl: i for i, lbl in enumerate(sorted_labels)}
        id2label = {i: lbl for lbl, i in label2id.items()}
        return label2id, id2label

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def get_split(self, split: str) -> list:
        splits = {"train": self.train, "dev": self.dev, "test": self.test}
        if split not in splits:
            raise ValueError(f"Split inválido '{split}'. Usa: train, dev, test")
        return splits[split]

    def get_tokens_and_labels(self, split: str) -> tuple:
        """
        Returns:
            tokens : list[list[str]]
            labels : list[list[str]]  (BIO)
        """
        sentences = self.get_split(split)
        tokens = [[tok for tok, _ in sent] for sent in sentences]
        labels = [[lbl for _, lbl in sent] for sent in sentences]
        return tokens, labels

    def get_label_names(self) -> list:
        return [self.id2label[i] for i in range(len(self.id2label))]

    def get_stats(self) -> dict:
        def n_entities(sents):
            return sum(1 for s in sents for _, lbl in s if lbl.startswith("B-"))
        def n_tokens(sents):
            return sum(len(s) for s in sents)

        return {
            "num_labels": len(self.label2id),
            "labels": self.get_label_names(),
            "train": {"sentences": len(self.train), "tokens": n_tokens(self.train), "entities": n_entities(self.train)},
            "dev":   {"sentences": len(self.dev),   "tokens": n_tokens(self.dev),   "entities": n_entities(self.dev)},
            "test":  {"sentences": len(self.test),  "tokens": n_tokens(self.test),  "entities": n_entities(self.test)},
        }

    # ------------------------------------------------------------------
    # BIO validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def validate_bio_format(labels: list) -> list:
        """
        Verifica que una secuencia BIO sea válida (no hay I- sin B- previo).

        Args:
            labels: Lista de etiquetas BIO de una oración.

        Returns:
            Lista de strings describiendo errores encontrados (vacía si OK).
        """
        errors = []
        prev_type = None
        for i, lbl in enumerate(labels):
            if lbl.startswith("I-"):
                entity_type = lbl[2:]
                if prev_type != entity_type:
                    errors.append(
                        f"Token {i}: I-{entity_type} sin B- previo "
                        f"(token anterior: {prev_type or 'O'})"
                    )
            prev_type = lbl[2:] if lbl.startswith(("B-", "I-")) else None
        return errors

    @staticmethod
    def validate_sentence_alignment(tokens: list, labels: list) -> list:
        """
        Verifica que tokens y labels tengan la misma longitud.

        Args:
            tokens: Lista de tokens.
            labels: Lista de etiquetas BIO.

        Returns:
            Lista de strings con errores (vacía si OK).
        """
        errors = []
        if len(tokens) != len(labels):
            errors.append(
                f"Desalineación: {len(tokens)} tokens vs {len(labels)} labels"
            )
        return errors

    def validate_split(self, split: str, verbose: bool = False) -> dict:
        """
        Ejecuta validaciones BIO sobre todas las oraciones de un split.

        Args:
            split: 'train', 'dev' o 'test'.
            verbose: Si True, imprime el resumen.

        Returns:
            Dict con 'total', 'ok', 'bio_errors', 'alignment_errors'.
        """
        sentences = self.get_split(split)
        tokens_list, labels_list = self.get_tokens_and_labels(split)

        bio_errors = 0
        align_errors = 0
        for tokens, labels in zip(tokens_list, labels_list):
            align_errs = self.validate_sentence_alignment(tokens, labels)
            bio_errs = self.validate_bio_format(labels)
            if align_errs:
                align_errors += len(align_errs)
            if bio_errs:
                bio_errors += len(bio_errs)

        result = {
            "total": len(sentences),
            "ok": len(sentences),
            "bio_errors": bio_errors,
            "alignment_errors": align_errors,
        }
        if verbose:
            print(
                f"[{split}] {result['total']} oraciones | "
                f"BIO errors: {bio_errors} | Alignment errors: {align_errors}"
            )
        return result

    def __repr__(self) -> str:
        s = self.get_stats()
        return (
            f"NERDataset(\n"
            f"  Labels: {s['num_labels']} {s['labels']}\n"
            f"  Train: {s['train']['sentences']} oraciones, {s['train']['tokens']:,} tokens, {s['train']['entities']} entidades\n"
            f"  Dev:   {s['dev']['sentences']} oraciones, {s['dev']['tokens']:,} tokens, {s['dev']['entities']} entidades\n"
            f"  Test:  {s['test']['sentences']} oraciones, {s['test']['tokens']:,} tokens, {s['test']['entities']} entidades\n"
            f")"
        )
