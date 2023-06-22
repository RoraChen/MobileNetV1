import numpy as np

def depthwise_conv(input, kernel_h, kernel_w, input_channel_d, padding, stride):
    # Get the number of input channels, height, and width of the input tensor
    input_channel, input_height, input_width = input.shape

    # Calculate the height and width of the output tensor
    output_height = (input_height + 2 * padding - kernel_h) // stride + 1
    output_width = (input_width + 2 * padding - kernel_w) // stride + 1

    # Create the output tensor
    output = np.zeros((input_channel_d, output_height, output_width))

    # If padding is required, pad the input tensor
    if padding > 0:
        input = np.pad(input, ((0, 0), (padding, padding), (padding, padding)), mode='constant')

    # Perform depthwise convolution
    for d in range(input_channel_d):
        for i in range(output_height):
            for j in range(output_width):
                # Extract the current receptive field
                receptive_field = input[:, i*stride:i*stride+kernel_h, j*stride:j*stride+kernel_w]

                # Perform element-wise multiplication, depthwise kernel multiplied by the receptive field
                output[d, i, j] = np.sum(receptive_field * depthwise_kernel[d])

    return output


def pointwise_conv(input, output_channels):
    # Get the number of input channels, height, and width of the input tensor
    input_channels, input_height, input_width = input.shape

    # Create the output tensor
    output = np.zeros((output_channels, input_height, input_width))

    # Perform pointwise convolution
    for d in range(output_channels):
        output[d] = np.sum(input * pointwise_kernel[d], axis=0)

    return output


# Define the architecture of MobileNetV1
layers = [
    # (Type, Kernel H, Kernel W, Kernel Channel D, Kernel Num K, Input R, Input C, Input Channel D, Padding, Stride)
    ("DW", 3, 3, 32, 1, 32, 32, 32, 0, 1),
    ("PW", 1, 1, 64, 32, 32, 32, 32, 0, 1),
    ("DW", 3, 3, 64, 1, 32, 32, 64, 1, 2),
    ("PW", 1, 1, 128, 64, 16, 16, 64, 0, 1),
    ("DW", 3, 3, 128, 1, 16, 16, 128, 1, 1),
    ("PW", 1, 1, 128, 128, 16, 16, 128, 0, 1),
    ("DW", 3, 3, 128, 1, 16, 16, 128, 1, 2),
    ("PW", 1, 1, 256, 128, 8, 8, 128, 0, 1),
    ("DW", 3, 3, 256, 1, 8, 8, 256, 1, 1),
    ("PW", 1, 1, 256, 256, 8, 8, 256, 0, 1),
    ("DW", 3, 3, 256, 1, 8, 8, 256, 1, 2),
    ("PW", 1, 1, 512, 256, 4, 4, 256, 0, 1),
    ("DW", 3, 3, 512, 1, 4, 4, 512, 1, 1),
    ("PW", 1, 1, 512, 512, 4, 4, 512, 0, 1),
    ("DW", 3, 3, 512, 1, 4, 4, 512, 1, 1),
    ("PW", 1, 1, 512, 512, 4, 4, 512, 0, 1),
    ("DW", 3, 3, 512, 1, 4, 4, 512, 1, 1),
    ("PW", 1, 1, 512, 512, 4, 4, 512, 0, 1),
    ("DW", 3, 3, 512, 1, 4, 4, 512, 1, 1),
    ("PW", 1, 1, 512, 512, 4, 4, 512, 0, 1),
    ("DW", 3, 3, 512, 1, 4, 4, 512, 1, 1),
    ("PW", 1, 1, 512, 512, 4, 4, 512, 0, 1),
    ("DW", 3, 3, 512, 1, 4, 4, 512, 1, 2),
    ("PW", 1, 1, 1024, 512, 2, 2, 512, 0, 1),
    ("DW", 3, 3, 1024, 1, 2, 2, 1024, 1, 1),
    ("PW", 1, 1, 1024, 1024, 2, 2, 1024, 0, 1)
]

# Execute the forward pass of MobileNetV1
input_tensor = np.random.rand(32, 32, 3)  # Input tensor
depthwise_kernel = np.random.rand(1024, 3, 3)  # Depthwise kernel tensor
pointwise_kernel = np.random.rand(1024, 1, 1)  # Pointwise kernel tensor

x = input_tensor
for layer in layers:
    if layer[0] == "DW":
        x = depthwise_conv(x, layer[1], layer[2], layer[7], layer[8], layer[9])
    elif layer[0] == "PW":
        x = pointwise_conv(x, layer[3])

print("Output tensor shape:", x.shape)
