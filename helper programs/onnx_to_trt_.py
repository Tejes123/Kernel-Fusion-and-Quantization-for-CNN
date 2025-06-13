# This code works for Resnet and CIFAR 10

import tensorrt as trt
import pycuda.driver as cuda

trtEngineName = "./ResnetModels/resnet18_fused.trt"

onnxPath = "./ResnetModels/resnet18_fused.onnx"

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, TRT_LOGGER)

with open(onnxPath, "rb") as model_file:
    if not parser.parse(model_file.read()):
        print("ERROR: Failed to parse ONNX model")
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise SystemExit(1)
print("✔️ ONNX parsed successfully")

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

# Enable FP16 if supported
if builder.platform_has_fast_fp16:
    config.set_flag(trt.BuilderFlag.FP16)

# Create optimization profile
profile = builder.create_optimization_profile()
input_name = network.get_input(0).name
profile.set_shape(input_name, (1, 3, 32, 32), (1, 3, 32, 32), (1, 3, 32, 32))
config.add_optimization_profile(profile)

print("Building engine…")
engine = builder.build_engine_with_config(network, config)
if engine is None:
    raise RuntimeError("❌ Engine build failed")

with open(trtEngineName, "wb") as f:
    f.write(engine.serialize())
print("✔️ Engine serialized to " + trtEngineName)