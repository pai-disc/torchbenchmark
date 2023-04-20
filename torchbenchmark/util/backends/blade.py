import argparse
from charset_normalizer import logging
try:
    import torch._dynamo as torchdynamo
    from torch._dynamo.backends.registry import register_backend as dynamo_backend
except ImportError:
    import torchdynamo
    from torchdynamo.optimizations.backends import create_backend as dynamo_backend

from torchbenchmark.util.backends import create_backend
import torch
import torch_blade
from torch_blade import optimize as blade_optimize
from torch_blade import mlir, tensorrt
from typing import List
import os

model_count = {}

def parse_blade_args(args) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # enable ofi by default
    parser.add_argument("--trt", action='store_true', help="use blade trt backend")
    parser.add_argument("--int8", action='store_true', help="whether to do quantization optimization")
    parser.add_argument("--cur_model", type=str, default='', help="current optimizing model")
    parser.add_argument("--reuse_model", action='store_true', help="whether to save/reuse blade optimized model")
    args = parser.parse_args(args)
    return args

@dynamo_backend
def blade_optimize_dynamo(subgraph, enable_fp16=False, use_trt=False, backend_args=[]):
    args = parse_blade_args(backend_args)
    optimized_model = None
    saved_model_path = ''
    cur_model = "dynamo-disc_" + args.cur_model
    global model_count
    if model_count.get(cur_model) == None:
        model_count[cur_model] = 1
    else:
        model_count[cur_model] += 1
    # for multi graph
    cur_model = cur_model + str(model_count[cur_model])
    if args.reuse_model and args.cur_model != '':
        _ = os.path.abspath(os.path.dirname(__file__))
        saved_model_dir = os.path.abspath(os.path.join(_, "../../../disc_saved_model"))
        saved_model_path = os.path.join(saved_model_dir, cur_model+".pb")
        if os.path.exists(saved_model_path):
            optimized_model = torch.jit.load(saved_model_path)
    if optimized_model == None:
        torch_config = torch_blade.config.Config()
        torch_config.enable_fp16 = enable_fp16
        if use_trt:
            torch_config.optimization_pipeline = torch_blade.tensorrt.backend_name()
        try:
            import torch._dynamo as torchdynamo
            with torch_config, torch.no_grad():
                optimized_model = blade_optimize(
                    subgraph.eval(),
                    allow_tracing=True,
                    model_inputs=tuple(data)
                )
        except ImportError:
            with torch_config, torch.no_grad():
                optimized_model = blade_optimize(
                    subgraph.model.eval(),
                    allow_tracing=True,
                    model_inputs=tuple(subgraph.example_inputs),
                )
    if use_trt:
        num_engines = tensorrt.num_engines
        num_compiled_nodes = tensorrt.num_compiled_nodes
    else:
        num_engines = mlir.num_engines
        num_compiled_nodes = mlir.num_compiled_nodes
    
    if num_engines(optimized_model) == 0:
        logging.warning("blade none fusion group")
    torchdynamo.utils.counters["blade"]["clusters"] += num_engines(optimized_model)
    torchdynamo.utils.counters["blade"]["compiled_nodes"] += sum(num_compiled_nodes(optimized_model))

    if args.reuse_model and not os.path.exists(saved_model_path):
        torch.jit.save(optimized_model, saved_model_path)

    # with open(f'model.code.py', 'a') as writer:
    #     writer.write(str(optimized_model.code))
    # with open(f'model.graph.txt', 'a') as writer:
    #     writer.write(str(optimized_model.graph))

    return optimized_model

@create_backend                   
def blade(model: 'torchbenchmark.util.model.BenchmarkModel', backend_args: List[str]):
    args = parse_blade_args(backend_args)
    optimized_model = None
    saved_model_path = ''
    cur_model = "disc_" + args.cur_model
    global model_count
    if model_count.get(cur_model) == None:
        model_count[cur_model] = 1
    else:
        model_count[cur_model] += 1
    # for multi graph
    cur_model = cur_model + str(model_count[cur_model])
    if args.reuse_model and args.cur_model != '':
        _ = os.path.abspath(os.path.dirname(__file__))
        saved_model_dir = os.path.abspath(os.path.join(_, "../../../disc_saved_model"))
        saved_model_path = os.path.join(saved_model_dir, cur_model+".pb")
        if os.path.exists(saved_model_path):
            optimized_model = torch.jit.load(saved_model_path)
    if optimized_model == None:
        module, example_inputs = model.get_module()
        torch_config = torch_blade.config.Config()
        torch_config.enable_fp16 = model.dargs.precision in ["fp16", "amp"]
        torch_config.enable_int8 = args.int8
        if args.trt:
            torch_config.optimization_pipeline = torch_blade.tensorrt.backend_name()
        with torch_config, torch.no_grad():
            optimized_model = blade_optimize(
                module.eval(),
                allow_tracing=True,
                model_inputs=tuple(example_inputs),
            )
    if isinstance(optimized_model, torch.jit.ScriptModule):
        for attr_name in getattr(model, "blade_reserve_attrs", []):
            attr = getattr(module, attr_name, None)
            if isinstance(attr, int):
                optimized_model._c._register_attribute(attr_name, torch._C.IntType.get(), attr)
            else:
                raise NotImplementedError(f"register_attribute for {attr_name} haven't been implemented")
    # with open(f'model.code.py', 'a') as writer:
    #     writer.write(str(optimized_model.code))
    # with open(f'model.graph.txt', 'a') as writer:
    #     writer.write(str(optimized_model.graph))

    if args.reuse_model and not os.path.exists(saved_model_path):
        torch.jit.save(optimized_model, saved_model_path)
    model.set_module(optimized_model)
