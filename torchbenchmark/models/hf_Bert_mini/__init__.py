from torchbenchmark.tasks import NLP
from torchbenchmark.util.framework.huggingface.model_factory import HuggingFaceModel

def prepare_fake_quant_model(model, example_inputs, extra_args):
    if '--trt' in extra_args:
        raise RuntimeError("Quantization only supports DISC backend.")

    from inspect import signature
    from torch_quant.quantizer import Backend, Quantizer
    from transformers.utils.fx import HFTracer

    class HFTracerWrapper(HFTracer):
        def __init__(self, input_names, **kwargs):
            super().__init__(**kwargs)
            self.input_names = input_names

        def trace(self, root, **kwargs):
            sig = signature(root.forward)
            concrete_args = {
                p.name: p.default for p in sig.parameters.values() if p.name not in self.input_names
            }
            return super().trace(root, concrete_args=concrete_args, **kwargs)

    # TODO: Generally speaking, attention_mask and token_type_ids are often passed in by the user
    # instead of using the default values. However torchbench's HuggingFaceModel only defines input_ids
    tracer = HFTracerWrapper(["input_ids"])
    quantizer = Quantizer(tracer=tracer, backend=Backend.DISC)
    calib_model = quantizer.calib(model)

    calib_model(
        **example_inputs
    )
    fake_quant_model = quantizer.quantize(model)
    return fake_quant_model


class Model(HuggingFaceModel):
    task = NLP.LANGUAGE_MODELING
    DEFAULT_TRAIN_BSIZE = 1
    DEFAULT_EVAL_BSIZE = 6

    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(name="hf_Bert_mini", test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)
        if 'blade' in extra_args and '--int8' in extra_args:
            # we use torch_quant to do quantization preprocess
            self.model = prepare_fake_quant_model(self.model, self.example_inputs, extra_args)
