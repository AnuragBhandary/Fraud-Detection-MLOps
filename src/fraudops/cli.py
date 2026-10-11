"""``fraudops <step>``: every stage of the project from one entry point."""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="fraudops")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare", help="raw Kaggle CSVs -> data/transactions.parquet + features")
    t = sub.add_parser("train", help="baseline, LightGBM, Optuna, calibration, MLflow registry")
    t.add_argument("--trials", type=int, default=30)
    sub.add_parser("leakage", help="random vs time split, future-peeking vs past-only features")
    sub.add_parser("build-store", help="snapshot the online feature store at the end of validation")
    sub.add_parser("drift", help="weekly PSI and performance report")
    b = sub.add_parser("bench", help="latency + online/offline parity through the HTTP API")
    b.add_argument("-n", type=int, default=5000)
    b.add_argument("--no-explain", action="store_true")
    s = sub.add_parser("serve", help="run the scoring API")
    s.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    from fraudops import pipeline

    if args.cmd == "prepare":
        from fraudops.data import write_parquet

        df = write_parquet()
        print(f"transactions: {df.shape}")
        pipeline.make_features()
    elif args.cmd == "train":
        pipeline.train(n_trials=args.trials)
    elif args.cmd == "leakage":
        pipeline.leakage()
    elif args.cmd == "build-store":
        pipeline.save_results("store", pipeline.build_store())
    elif args.cmd == "drift":
        pipeline.drift()
    elif args.cmd == "bench":
        from fraudops import bench

        name = "bench_no_explain" if args.no_explain else "bench"
        pipeline.save_results(name, bench.run(n=args.n, explain=not args.no_explain))
    elif args.cmd == "serve":
        import uvicorn

        uvicorn.run("fraudops.serve:app", host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
