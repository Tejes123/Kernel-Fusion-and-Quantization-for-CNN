import tensorrt as trt
batch_size = 64 

onnxPath = r"C:\Tejeswar\Fusion\resnet50_new.onnx"

TRT_LOGGER = trt.Logger()
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

for i in range(network.num_inputs):
    inp = network.get_input(i)
    print(f"INPUT  '{inp.name}': {inp.shape}")

for i in range(network.num_outputs):
    out = network.get_output(i)
    print(f"OUTPUT '{out.name}': {out.shape}")