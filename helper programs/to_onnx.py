import torch
import torch.nn as nn
import torchvision

output_onnx="resnet50_fused_new.onnx"

PATH = r"C:\Tejeswar\Fusion\ResnetModels\resnet50_fused_new.pth"

model  = torchvision.models.resnet50()

in_features = model.fc.in_features

# Replace the last fully connected layer with a new one for 10 classes
model.fc = nn.Linear(in_features, 10)

model.load_state_dict(torch.load(PATH, weights_only=True))
model.eval()

batch_size = 16
# Generate input tensor with random values
input_tensor = torch.rand(batch_size, 3, 32, 32)

# Export torch model to ONNX
print("Exporting ONNX model {}".format(output_onnx))
torch.onnx.export(model, input_tensor, output_onnx,
    opset_version=12,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"},
                  "output": {0: "batch", 2: "height", 3: "width"}},
    verbose=False)