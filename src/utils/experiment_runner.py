"""
experiment_runner.py

Grid search sobre modelos × text_modes × cleaning_types para clasificación multilabel.
Basado en el ExperimentRunner de Building-Open-Datasets-for-Spanish-Medical-Tasks.

Uso:
    from utils.experiment_runner import ExperimentRunner

    runner = ExperimentRunner(
        data_dir="data/pubmed_mesh",
        output_base="results/multilabel",
        models=["dccuchile/bert-base-spanish-wwm-cased"],
        text_modes=["title_abstract", "abstract_only"],
        filter_geographicals_options=[False, True],
        num_epochs=3,
        batch_size=8,
        learning_rate=2e-5,
        wandb_project="spanish-multilabel",
    )
    runner.run()
"""

import json
import os
from pathlib import Path
from typing import List, Optional


class ExperimentRunner:
    """
    Ejecuta una cuadrícula de experimentos para clasificación multilabel.

    Por cada combinación (model, text_mode, filter_geographicals) entrena
    un EncoderMultiLabelModel y guarda los resultados.
    """

    def __init__(
        self,
        data_dir: str,
        output_base: str = "results/multilabel",
        models: Optional[List[str]] = None,
        text_modes: Optional[List[str]] = None,
        filter_geographicals_options: Optional[List[bool]] = None,
        num_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        max_length: int = 512,
        threshold: float = 0.5,
        wandb_project: Optional[str] = None,
    ):
        self.data_dir = data_dir
        self.output_base = output_base
        self.models = models or ["dccuchile/bert-base-spanish-wwm-cased"]
        self.text_modes = text_modes or ["title_abstract"]
        self.filter_geo_options = filter_geographicals_options or [False]
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.threshold = threshold
        self.wandb_project = wandb_project

        self.results: List[dict] = []

    def run(self) -> List[dict]:
        """
        Ejecuta la grilla completa.

        Returns:
            Lista de dicts con config + métricas por experimento.
        """
        configs = [
            (model, text_mode, filter_geo)
            for model in self.models
            for text_mode in self.text_modes
            for filter_geo in self.filter_geo_options
        ]

        print(f"\n{'='*60}")
        print(f"ExperimentRunner: {len(configs)} experimentos")
        print(f"{'='*60}\n")

        for i, (model_name, text_mode, filter_geo) in enumerate(configs, 1):
            tag = f"[{i}/{len(configs)}]"
            print(f"\n{tag} model={model_name.split('/')[-1]} | text_mode={text_mode} | filter_geo={filter_geo}")

            result = self._run_single(model_name, text_mode, filter_geo)
            self.results.append(result)

        self._save_summary()
        return self.results

    def _run_single(
        self, model_name: str, text_mode: str, filter_geo: bool
    ) -> dict:
        """Entrena y evalúa un único experimento."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))

        from utils.multilabel_datareader import MultiLabelDataset
        from models.encoder_multilabel import EncoderMultiLabelModel
        from evaluation.multilabel_predictor import MultiLabelPredictor

        geo_tag = "no_geo" if filter_geo else "all"
        model_short = model_name.split("/")[-1]
        output_dir = os.path.join(
            self.output_base, f"{model_short}_{text_mode}_{geo_tag}"
        )

        dataset = MultiLabelDataset(
            data_dir=self.data_dir,
            text_mode=text_mode,
            filter_geographicals=filter_geo,
        )

        wandb_run = self._init_wandb(model_name, text_mode, filter_geo, dataset)

        model = EncoderMultiLabelModel(
            model_name=model_name,
            label2id=dataset.label2id,
            id2label=dataset.id2label,
            max_length=self.max_length,
            threshold=self.threshold,
        )

        train_texts, train_labels = dataset.get_texts_and_labels("train")
        dev_texts, dev_labels = dataset.get_texts_and_labels("dev")

        model.train(
            train_texts=train_texts,
            train_labels=train_labels,
            dev_texts=dev_texts,
            dev_labels=dev_labels,
            output_dir=output_dir,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
        )

        predictor = MultiLabelPredictor(model, dataset)
        predictor.save_predictions(split="test", output_path=output_dir)

        metrics = {}
        metrics_path = Path(output_dir) / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)

        if wandb_run:
            self._log_and_finish_wandb(wandb_run, metrics)

        result = {
            "model": model_name,
            "text_mode": text_mode,
            "filter_geographicals": filter_geo,
            "output_dir": output_dir,
            **metrics,
        }
        print(f"  -> F1 micro: {metrics.get('f1_micro', 'N/A'):.4f}" if "f1_micro" in metrics else "  -> metrics not available")
        return result

    def _init_wandb(self, model_name, text_mode, filter_geo, dataset):
        if not self.wandb_project:
            return None
        try:
            import wandb
            run = wandb.init(
                project=self.wandb_project,
                name=f"{model_name.split('/')[-1]}_{text_mode}_{'no_geo' if filter_geo else 'all'}",
                config={
                    "model_name": model_name,
                    "text_mode": text_mode,
                    "filter_geographicals": filter_geo,
                    "num_labels": len(dataset.label2id),
                    "train_size": len(dataset.train),
                    "num_epochs": self.num_epochs,
                    "batch_size": self.batch_size,
                    "learning_rate": self.learning_rate,
                },
                reinit=True,
            )
            return run
        except ImportError:
            return None

    @staticmethod
    def _log_and_finish_wandb(wandb_run, metrics: dict) -> None:
        wandb_run.log(metrics)
        wandb_run.finish()

    def _save_summary(self) -> None:
        """Guarda un resumen JSON de todos los experimentos."""
        summary_path = Path(self.output_base) / "experiment_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"\nResumen guardado en: {summary_path}")

    def print_summary(self) -> None:
        """Imprime tabla de resultados."""
        if not self.results:
            print("Sin resultados.")
            return

        header = f"{'Model':<40} {'TextMode':<20} {'FilterGeo':<10} {'F1 Micro':<10} {'F1 Macro':<10}"
        print("\n" + "=" * len(header))
        print(header)
        print("=" * len(header))
        for r in self.results:
            model_short = r["model"].split("/")[-1][:38]
            f1_micro = f"{r['f1_micro']:.4f}" if "f1_micro" in r else "N/A"
            f1_macro = f"{r.get('f1_macro', 'N/A')}"
            if isinstance(f1_macro, float):
                f1_macro = f"{f1_macro:.4f}"
            print(
                f"{model_short:<40} {r['text_mode']:<20} {str(r['filter_geographicals']):<10} "
                f"{f1_micro:<10} {f1_macro:<10}"
            )
        print("=" * len(header))
