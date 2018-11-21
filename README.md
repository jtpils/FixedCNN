### Fixed-point Convolutional Neural Networks implemented by MATLAB

The motivation for this project is creating a genearal library which can simulate a CNN with fixed-point operations. Since python is poor in fixed-point calculation support, I try to write the project in MATLAB thoroughly. While the fi object in MATLAB can conviniently express fixed-point(FP) operations, there are also many functions in MATLAB that don't support fi. So I have to rewrite a lot of basic functions such as conv2d etc.. step by step.

### Completed
Several fundamental functions have been completed and carried on in a parallel way as possible. The comprehensive review of all functions as below:

|   **Type**  |    **Status**   |       **Description**       |
|:-------:|:-----------:|:-----------------------:|
| Conv2d  | Completed   |                         |
| Pooling | Uncompleted | Only support MAX method |
| ReLU    | Completed   |                         |
| FC      | Uncompleted | Working on it           |

### Requirement

All codes are tested in MATLAB R2017b and don't support GPU acceleration, which means you should only run it on small dataset otherwise the runtime will be quite scaring. As I say above, this library is designed for simulating FP-CNN and it's an experimental research which helps us understand FP behaviors of deep neural networks. Furthermore, it can help people who want to deploy their CNN algorithms on FP devices (FPGA/ASIC etc) verify the effectiveness of quantization method.

### TODO

Although this library can implement a simple CNN with FP, there many problems needed to solve.
