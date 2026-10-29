from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .collector import LotteryDataCollector, build_metadata_snapshot
from .modeling import LotteryForecaster


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pipeline de loterías para Colombia.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser("collect", help="Extrae datos y guarda CSV.")
    collect_parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/lottery_results.csv"),
        help="Ruta de salida para datos normalizados.",
    )
    collect_parser.add_argument(
        "--limit",
        type=int,
        default=50000,
        help="Límite de registros por fuente.",
    )
    collect_parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("data/raw/metadata.json"),
        help="Ruta de salida para metadatos de fuentes.",
    )

    train_parser = subparsers.add_parser(
        "train", help="Entrena modelos por estrategia y genera predicción."
    )
    train_parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/lottery_results.csv"),
        help="CSV de entrada con datos normalizados.",
    )
    train_parser.add_argument(
        "--models-output",
        type=Path,
        default=Path("models/lottery_models.pkl"),
        help="Archivo de salida para modelos serializados.",
    )
    train_parser.add_argument(
        "--predictions-output",
        type=Path,
        default=Path("predictions/daily_predictions.csv"),
        help="CSV de salida para predicciones diarias.",
    )
    train_parser.add_argument(
        "--prediction-date",
        type=str,
        default=None,
        help="Fecha a predecir (YYYY-MM-DD). Por defecto usa hoy UTC.",
    )
    train_parser.add_argument(
        "--min-rows",
        type=int,
        default=30,
        help="Mínimo de filas por estrategia/juego para entrenar.",
    )

    return parser


def collect_command(output: Path, metadata_output: Path, limit: int) -> None:
    collector = LotteryDataCollector()
    df = collector.collect(limit=limit)
    if df.empty:
        raise RuntimeError("No se obtuvieron datos válidos desde las fuentes configuradas.")
    collector.save_to_csv(df, output)
    build_metadata_snapshot(metadata_output)
    print(f"Datos guardados en: {output}")


def train_command(
    input_path: Path,
    models_output: Path,
    predictions_output: Path,
    prediction_date: str | None,
    min_rows: int,
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"No existe el archivo de entrada: {input_path}")

    df = pd.read_csv(input_path)
    if df.empty:
        raise RuntimeError("El archivo de entrada no contiene datos.")

    df["draw_date"] = pd.to_datetime(df["draw_date"], errors="coerce")
    df["number"] = pd.to_numeric(df["number"], errors="coerce")
    df = df.dropna(subset=["draw_date", "number", "strategy", "game"])
    if df.empty:
        raise RuntimeError("No hay datos válidos para entrenar.")

    forecaster = LotteryForecaster(min_rows=min_rows)
    models = forecaster.train_models(df)
    if not models:
        raise RuntimeError(
            "No se entrenaron modelos. Verifica cantidad de datos y columnas de estrategia/juego."
        )

    forecast_date = pd.to_datetime(prediction_date) if prediction_date else None
    predictions = forecaster.predict_next_day(models, prediction_date=forecast_date)

    LotteryForecaster.save_models(models, models_output)
    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_output, index=False)
    print(f"Modelos guardados en: {models_output}")
    print(f"Predicciones guardadas en: {predictions_output}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "collect":
        collect_command(output=args.output, metadata_output=args.metadata_output, limit=args.limit)
        return

    train_command(
        input_path=args.input,
        models_output=args.models_output,
        predictions_output=args.predictions_output,
        prediction_date=args.prediction_date,
        min_rows=args.min_rows,
    )


if __name__ == "__main__":
    main()

