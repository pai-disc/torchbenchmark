import intel_extension_for_pytorch as ipex
from torchbenchmark.util.backends import create_backend
import torch
from typing import List

@create_backend                   
def ipex_script(model: 'torchbenchmark.util.model.BenchmarkModel', backend_args: List[str]):
    try:
        import intel_extension_for_pytorch as ipex
    except ImportError:
        return
    module, data = model.get_module()
    if model.dargs.precision == "fp32":
        optimized_model = ipex.optimize(module.eval(), dtype=torch.float32)
    elif model.dargs.precision == "amp" or model.dargs.precision == "bfloat16":
        optimized_model = ipex.optimize(module.eval(), dtype=torch.bfloat16)
    elif model.dargs.precision == "int8-static":
        from intel_extension_for_pytorch.quantization import prepare, convert
        qconfig = ipex.quantization.default_static_qconfig
        ## TODO: fix len > 1
        if len(data) == 1:
            data = data[0]
        prepared_model = prepare(module.eval(), qconfig, example_inputs=data, inplace=False)
        n_iter = 100
        for i in range(n_iter):
            prepared_model(data)
        optimized_model = convert(prepared_model)
    elif model.dargs.precision == "int8-dynamic":
        from intel_extension_for_pytorch.quantization import prepare, convert
        qconfig = ipex.quantization.default_dynamic_qconfig
        prepared_model = prepare(module.eval(), qconfig, example_inputs=data)
        optimized_model = convert(prepared_model)
    
    with torch.no_grad(), torch.cpu.amp.autocast(enabled=(model.dargs.precision=='bfloat16' or model.dargs.precision=="amp")):
        optimized_model = torch.jit.trace(optimized_model, data, check_trace=False, strict=False)
        optimized_model = torch.jit.freeze(optimized_model)

    if isinstance(optimized_model, torch.jit.ScriptModule):
        for attr_name in getattr(model, "blade_reserve_attrs", []):
            attr = getattr(module, attr_name, None)
            if isinstance(attr, int):
                optimized_model._c._register_attribute(attr_name, torch._C.IntType.get(), attr)
            else:
                raise NotImplementedError(f"register_attribute for {attr_name} haven't been implemented")

    model.set_module(optimized_model)

