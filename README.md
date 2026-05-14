## Paper

This repository is the official implementation of:

**[QuEENet: Quantum-Enhanced Expressive Network for Image Classification](https://openaccess.thecvf.com/content/WACV2026/papers/Bayal_QuEENet_Quantum-Enhanced_Expressive_Network_for_Image_Classification_WACV_2026_paper.pdf)**  
Shashank Bayal*, Rushikesh Govind Dawane*, Komal*, Santosh Kumar Vipparthi, Subrahmanyam Murala  
*WACV 2026*

> QuEENet is a hybrid quantum-classical architecture for image classification that integrates parameterized quantum circuits with non-Clifford gates into a CNN backbone. With only **0.085M parameters**, it achieves competitive accuracy across CIFAR-10 (85.73%), MNIST (99.11%), Fashion-MNIST (92.70%), and Medical-MNIST (99.99%).

##Procedure
1. Install python libraries required for this Queenet code using requirement.txt file provided in repo.
2. Give input path of the dataset folder (change number of output classes as per the requirements)
3. In terminal run command " python Queenet_n_fashion_mnist.py"


### Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{bayal2026queenet,
  title={QuEENet: Quantum-Enhanced Expressive Network for Image Classification},
  author={Bayal, Shashank and Dawane, Rushikesh Govind and Komal, Komal and Vipparthi, Santosh Kumar and Murala, Subrahmanyam},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={7883--7892},
  year={2026}
}
```
