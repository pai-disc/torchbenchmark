"""
Support TorchDynamo(https://github.com/facebookresearch/torchdynamo) backends
"""
import argparse
import functools
from typing import List
import torchdynamo
from .blade import blade_optimize_dynamo


TORCHDYNAMO_ROUNDS = 3

def parse_torchdynamo_args(model: 'torchbenchmark.util.model.BenchmarkModel', dyamo_args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    available_backends = torchdynamo.list_backends()
    parser.add_argument(
        "--torchdynamo", choices=available_backends, help="Specify torchdynamo backends"
    )
    parser.add_argument(
        "--trt", action='store_true', help="use blade trt backend"
    )
    args = parser.parse_args(dyamo_args)
    return args


def apply_torchdynamo_args(model: 'torchbenchmark.util.model.BenchmarkModel', args: argparse.Namespace, precision: str):
    torchdynamo.config.raise_on_backend_error = False
    torchdynamo.reset()
    torchdynamo.utils.counters.clear()

    if args.torchdynamo == "fx2trt" and precision == "fp16":
        model.add_context(functools.partial(torchdynamo.optimize, torchdynamo.optimizations.backends.fx2trt_compiler_fp16))
    elif "blade" in args.torchdynamo:
        model.add_context(functools.partial(torchdynamo.optimize, \
            functools.partial(blade_optimize_dynamo, enable_fp16=precision=="fp16", use_trt=args.trt)))
    else:
        model.add_context(functools.partial(torchdynamo.optimize, args.torchdynamo))
    
    for _ in range(TORCHDYNAMO_ROUNDS):
        model.invoke()
    model.run_contexts.pop()
    model.add_context(torchdynamo.run)
