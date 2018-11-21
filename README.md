### Fixed-point Convolutional Neural Networks implemented by MATLAB

The motivation for this project is creating a genearal library which can simulate a CNN with fixed-point operations. Since python is poor in fixed-point calculation support, I try to write the project in MATLAB thoroughly. While the fi object in MATLAB can convinently express fixed-point(FP) operations, there are also many functions in MATLAB that don't support fi. So I have to rewrite a lot of basic functions such as conv2d etc.. step by step.

### Completed

Several fundamental functions have been completed and carried on in a parallel way as possible.

#### Conv2d 

#### Pooling2d

#### ReLU

### Ongoing

Although this library can implement a simple CNN with FP, there many problems needed to solve.
