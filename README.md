### Fixed-point Convolutional Neural Networks Simulation Library in MATLAB

The motivation for this project is creating a genearal library which can simulate a CNN with fixed-point operations. Since python is poor in fixed-point calculation support, I try to write the project in MATLAB thoroughly. While the fi object in MATLAB can conviniently express fixed-point(FP) operations, there are still many functions in MATLAB that don't support fi. So I have to rewrite a lot of basic functions such as conv2d etc step by step.

### Project Progress
Several fundamental functions have been completed and carried on in a parallel way as possible. The comprehensive review of all functions as below:

|        **Type**         |       **Status**     |        **Description**       |
|:-----------------------:|:--------------------:|:----------------------------:|
| Conv2d                  |      Completed       |                              |
| Depthwise Conv          |     Uncompleted      |        Working on it         |
| Pooling                 |      Completed       |           MAX/AVG*           |
| ReLU                    |      Completed       |                              |
| FC                      |     Uncompleted      |        Working on it         |
| Softmax                 |     Uncompleted      |        Working on it         |
#### * Now you can use self-defined pooling function other than MAX/AVG. Before applying customed pooling function, you should add the function name and its definition in the Pool_Type register table.

#### Some Details

- The parallelism level (PL) of the source code is from vector to tensor. The definition of different levels is as below:
  - L1 Vector: 1×N array
  - L2 Matrix: M×N matrix
  - L3 Tensor: M×N×[H1,H2 ...], where the length of [H] is no less than 1.
  
  Higher the PL is, faster the function runs.

- Conv2d calculates 2d convolution of the input tensor, PL is L3, the input and output format are TF-compatible.
- Pooling calculates 2d pooling of the input tensor, PL is L3, while pooling function doesn't support 3d pooling like TF.

### Requirement

All codes are tested in **MATLAB R2017b** and don't support GPU acceleration, which means you should only run it on small dataset otherwise the runtime will be quite scaring. As I say above, this library is designed for simulating FP-CNN and it's also an experimental research which helps us understand FP behaviors of deep neural networks. Furthermore, it can help people who want to deploy their CNN algorithms on FP devices (FPGA/ASIC etc) to verify the effectiveness of quantization method.

### TODO

- Although the MATLAB library can implement a simple FP-CNN for now, there are still many problems in this code. I will fix these problems in the future.

- To enhance the library's robustness and compatibility, an elaborate and overall unit test module is under construsting which will check more complicated conditions in application scenarios.

- I try my best to design every function to be similar to TensorFlow-style as possible so that people who are familiar with TensorFlow can easily tranfer to this library without much learning effort.
