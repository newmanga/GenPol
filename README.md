# GenPol

## Overview
This project explores how diffusion generative models and Generative Adversarial Networks (GANs) can be combined to improve the performance of Inverse Reinforcement Learning (IRL) algorithms. While GANs can generate high-quality data, they often suffer from poor mode coverage, meaning they fail to capture the diversity of the data distribution. Diffusion models, on the other hand, produce high-quality samples but are computationally expensive to sample from. Our approach aims to leverage the strengths of both techniques to create a model that achieves efficient sampling, high-quality data generation, and improved sample diversity.

The key idea is to use a forward diffusion process to smooth the data distribution learned by the GAN's discriminator, thereby enhancing generalization and improving mode coverage. This approach is inspired by prior work on Gaussian-mixture distributed instance noise, which can be seen as a form of data augmentation.

## Key Features
- **Efficient Sampling**: Leverages the efficient sampling capabilities of GANs while incorporating diffusion processes to improve diversity.
- **High-Quality Samples**: Achieves high-quality sample generation, comparable to or surpassing GANs alone.
- **Improved Sample Diversity**: The forward diffusion process addresses the mode collapse issue often seen in GANs.

## Implementation Details
1. **Baseline**: The Generative Adversarial Imitation Learning (GAIL) algorithm is implemented as the starting point.
2. **Forward Diffusion Process**: A forward diffusion process, inspired by prior work on diffusion models, is added to smooth the data distribution.
3. **Environment**: Experiments are conducted in the HalfCheetah environment using the Mujoco simulator.

### Model Architecture
- **Discriminator, Value, and Generator Networks**: All networks follow a 3-layer multi-layer perceptron (MLP) architecture with 50 hidden units in each layer and ReLU activations.
- **Optimizer**: The discriminator network is trained using the Adam optimizer with a learning rate of 1e-3.
- **Training**: The policy network is trained using the TRPO (Trust Region Policy Optimization) algorithm with causal entropy regularization, as done in the original GAIL study.
- **Diffusion Parameters**: The update rate and maximum value of the adaptive diffusion process are set to 1 and 1000, respectively.

## Evaluation
The approach was evaluated using expert demonstrations obtained from a policy trained with TRPO in the HalfCheetah environment. The following metrics were used to assess the model's performance:
- **Discriminator Probabilities**: Measures the convergence speed of the discriminator in distinguishing between generated and expert data.
- **Cumulative Rewards**: Compares the rewards achieved by the GAIL baseline and the GAIL+diffusion model.

The results demonstrate that incorporating the diffusion process accelerates convergence of the discriminator's probabilities while maintaining cumulative rewards that are comparable to those of the expert policy. The enhanced generalization capabilities of the discriminator are evident from these results.

## Results
The key results are visualized in the following plots:
1. **Discriminator Probabilities**: The convergence of the discriminator's probabilities is faster when diffusion is incorporated.
2. **Cumulative Rewards**: The GAIL+diffusion approach achieves cumulative rewards comparable to the expert policy, demonstrating the effectiveness of the combined approach.

## References
1. Ho, J., & Ermon, S. (2016). Generative Adversarial Imitation Learning (GAIL).
2. Wang, Z., et al. (2023). DIFFUSION-GAN: TRAINING GANS WITH DIFFUSION.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

