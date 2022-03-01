# OverlapTransformer

The code for our paper submitted to IROS 2022:  

**OverlapTransformer: An Efficient and Rotation-Invariant Transformer Network for LiDAR-Based Place Recognition.**  

OverlapTransformer is a novel lightweight neural network exploiting the range image representation of LiDAR sensors to achieve fast execution with less than 4 ms per frame.  

<img src="https://github.com/haomo-ai/OverlapTransformer/blob/master/query_database.gif" >  

This is an animation for finding the top1 candidate with **OverlapTransformer** on sequence 1-1 (database) and 1-3 (query) of **Haomo dataset**. 

<img src="https://github.com/haomo-ai/OverlapTransformer/blob/master/haomo_dataset.pdf" >  

**Haomo dataset** which is collected by **HAOMO.AI** will be released soon.   


## Dependencies

We are using pytorch-gpu for neural networks.

A nvidia GPU is needed for faster retrival.
OverlapTransformer (OT) is also fast enough when using the neural network on CPU.

To use a GPU, first you need to install the nvidia driver and CUDA.

- CUDA Installation guide: [link](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

- System dependencies:

  ```bash
  sudo apt-get update 
  sudo apt-get install -y python3-pip python3-tk
  sudo -H pip3 install --upgrade pip
  ```

- Python dependencies (may also work with different versions than mentioned in the requirements file)

  ```bash
  sudo -H pip3 install -r requirements.txt
  ```
