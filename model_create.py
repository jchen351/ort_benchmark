import onnx
from onnx import helper, TensorProto

# Define the input tensor (N, C, H, W)
input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])

# Define the output tensor
output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 16, 222, 222])

# Create a Conv layer
conv_weight = helper.make_tensor(
    name="conv_weight",
    data_type=TensorProto.FLOAT,
    dims=[16, 3, 3, 3],  # Output channels, input channels, kernel height, kernel width
    vals=[1.0] * (16 * 3 * 3 * 3),  # Initialize with ones
)
conv_node = helper.make_node(
    "Conv",  # Name of the operation
    inputs=["input", "conv_weight"],  # Inputs: input tensor, weight tensor
    outputs=["conv_output"],  # Output: conv_output tensor
    kernel_shape=[3, 3],  # 3x3 kernel
    pads=[0, 0, 0, 0],  # No padding
    strides=[1, 1],  # Stride 1x1
)

# Create a Relu layer
relu_node = helper.make_node(
    "Relu",  # Name of the operation
    inputs=["conv_output"],  # Input: conv_output from Conv layer
    outputs=["output"],  # Output: final output tensor
)

# Create the graph
graph_def = helper.make_graph(
    nodes=[conv_node, relu_node],  # List of operations in the graph
    name="SimpleConvReluModel",  # Name of the graph
    inputs=[input_tensor],  # Inputs to the model
    outputs=[output_tensor],  # Outputs of the model
    initializer=[conv_weight],  # Initializers (weights)
)

# Create the model
opset_version = 18
model_def = helper.make_model(graph_def, producer_name="onnx-example", opset_imports=[helper.make_opsetid("", opset_version)])
model_def.ir_version = 8 # Set the IR version
onnx.checker.check_model(model_def)  # Verify the model is valid
onnx.save(model_def, "simple_conv_relu.onnx")  # Save the model to a file

print("ONNX model saved as 'simple_conv_relu.onnx'")