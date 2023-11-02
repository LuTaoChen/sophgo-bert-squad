import os
import sys
import argparse
from dataloder import get_dataloader
from eval import evaluator

from bert_runner import get_runner

sys.path.insert(0, os.getcwd())

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../../dataset/squad/dev-v1.1.json", help="data file path")
    parser.add_argument("--scenario", choices=["SingleStream", "Offline", "Server", "MultiStream"], default="Offline", help="Scenario")
    parser.add_argument("--accuracy", action="store_true", help="enable accuracy pass")
    parser.add_argument("--count", type=int, default=50, help="Maximum number of examples to consider")
    parser.add_argument("--model", type=str, help="bmodel path")
    parser.add_argument("--cache_path", default='eval_features.pickle', help="pickle path")
    parser.add_argument("--out_file", default='./result/predictions.json', help='output json path')

    args = parser.parse_args()
    return args

def main():
    if not os.path.exists("result"):
        os.makedirs("result")
    args = get_args()

    runner = get_runner(args)
    # warmup
    # runner.run_one_item()

    config = {
        "accuracy": args.accuracy,
        "total_count": args.count,
    }
    squad_dl = get_dataloader(count_override=args.count,
                              cache_path=args.cache_path,
                              input_file=args.data,
                              load_fn=runner.run_one_query)
    squad_dl.start_test(config)
    result = runner.runner_result
    if args.accuracy:
        evaluator(args, result)
    print("Done!")

if __name__ == "__main__":
    main()
