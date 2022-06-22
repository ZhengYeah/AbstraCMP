import torch
import torch.onnx
from mnist_net import NeuralNetwork

def pth_to_onnx(input, checkpoint, onnx_path, input_names=['input'], output_names=['output'], device='cpu'):
    if not onnx_path.endswith('.onnx'):
        print('Warning! The onnx model name is not correct,\
              please give a name that ends with \'.onnx\'!')
        return
    model = NeuralNetwork()
    model.load_state_dict(torch.load(checkpoint))
    model.eval()
    torch.onnx.export(model, input, onnx_path, verbose=True, input_names=input_names, output_names=output_names)
    print("Exporting .pth model to onnx model has been successful!")


if __name__ == '__main__':
    checkpoint = 'checkpoint/mnist_net.pth'
    onnx_path = 'checkpoint/mnist_net_gradient.onnx'
    # dummy input: (tensor) size of [minibatch size, input size]
    dummy_input = torch.randn([1, 784])
    pth_to_onnx(dummy_input, checkpoint, onnx_path)
