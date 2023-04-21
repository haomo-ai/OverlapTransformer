# OverlapTransformer

The code for our paper for **RAL/IROS 2022**:  

**OverlapTransformer: An Efficient and Yaw-Angle-Invariant Transformer Network for LiDAR-Based Place Recognition.**  [[paper](https://arxiv.org/pdf/2203.03397.pdf)]

OverlapTransformer (OT) is a novel lightweight neural network exploiting the LiDAR range images to achieve fast execution with **less than 4 ms per frame using python, less than 2 ms per frame using C++** in LiDAR similarity estimation. 
It is a newer version of our previous [OverlapNet](https://github.com/PRBonn/OverlapNet), which is faster and more accurate in LiDAR-based loop closure detection and place recognition.

Developed by [Junyi Ma](https://github.com/BIT-MJY), [Xieyuanli Chen](https://github.com/Chen-Xieyuanli) and [Jun Zhang](https://github.com/zhangjun-xyz).

## News!

**[2022-12]** SeqOT is accepted by IEEE Transactions on Industrial Electronics (TIE)!  
**[2022-09]** We further develop **a sequence-enhanced version of OT named as SeqOT**, which can be found [here](https://github.com/BIT-MJY/SeqOT).

## Haomo Dataset

<img src="https://github.com/haomo-ai/OverlapTransformer/blob/master/query_database_haomo.gif" >  

Fig. 1 An online demo for finding the top1 candidate with OverlapTransformer on sequence 1-1 (database) and 1-3 (query) of [Haomo Dataset](https://github.com/haomo-ai/OverlapTransformer/tree/master/Haomo_Dataset).

<div align=center>
<img src="https://github.com/haomo-ai/OverlapTransformer/blob/master/Haomo_Dataset/haomo_dataset.png" width="98%"/>
</div>  

Fig. 2 [Haomo Dataset](https://github.com/haomo-ai/OverlapTransformer/tree/master/Haomo_Dataset) which is collected by **HAOMO.AI**. 

More details of Haomo Dataset can be found in dataset description ([link](https://github.com/haomo-ai/OverlapTransformer/tree/master/Haomo_Dataset)).

## Table of Contents
1. [Introduction and Haomo Dataset](#OverlapTransformer)
2. [Publication](#Publication)
3. [Dependencies](#Dependencies)
4. [How to Use](#How-to-Use)
5. [Datasets Used by OT](#Datasets-Used-by-OT)
6. [Related Work](#Related-Work)
7. [License](#License)

## Publication

If you use the code or the Haomo dataset in your academic work, please cite our paper ([PDF](https://arxiv.org/pdf/2203.03397.pdf)):

```
@ARTICLE{ma2022ral,
  author={Ma, Junyi and Zhang, Jun and Xu, Jintao and Ai, Rui and Gu, Weihao and Chen, Xieyuanli},
  journal={IEEE Robotics and Automation Letters}, 
  title={OverlapTransformer: An Efficient and Yaw-Angle-Invariant Transformer Network for LiDAR-Based Place Recognition}, 
  year={2022},
  volume={7},
  number={3},
  pages={6958-6965},
  doi={10.1109/LRA.2022.3178797}}
```

## Dependencies

We use pytorch-gpu for neural networks.

An nvidia GPU is needed for faster retrival.
OverlapTransformer is also fast enough when using the neural network on CPU.

To use a GPU, first you need to install the nvidia driver and CUDA.

- CUDA Installation guide: [link](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)  
  We use CUDA 11.3 in our work. Other versions of CUDA are also supported but you should choose the corresponding torch version in the following Torch dependences.  

- System dependencies:

  ```bash
  sudo apt-get update 
  sudo apt-get install -y python3-pip python3-tk
  sudo -H pip3 install --upgrade pip
  ```
- Torch dependences:  
  Following this [link](https://pytorch.org/get-started/locally/), you can download Torch dependences by pip:
  ```bash
  pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
  ```
  or by conda:
  ```bash
  conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
  ```  
  

- Other Python dependencies (may also work with different versions than mentioned in the requirements file):

  ```bash
  sudo -H pip3 install -r requirements.txt
  ```
  
## How to Use
We provide a training and test tutorials for KITTI sequences in this repository. 
The tutorials for Haomo dataset will be released together with the complete Haomo dataset.  

We recommend you follow our code and data structures as follows.

### Code Structure

```bash
├── config
│   ├── config_haomo.yml
│   └── config.yml
├── modules
│   ├── loss.py
│   ├── netvlad.py
│   ├── overlap_transformer_haomo.py
│   └── overlap_transformer.py
├── test
│   ├── test_haomo_topn_prepare.py
│   ├── test_haomo_topn.py
│   ├── test_kitti00_prepare.py
│   ├── test_kitti00_PR.py
│   ├── test_kitti00_topN.py
│   ├── test_results_haomo
│   │   └── predicted_des_L2_dis_bet_traj_forward.npz (to be generated)
│   └── test_results_kitti
│       └── predicted_des_L2_dis.npz (to be generated)
├── tools
│   ├── read_all_sets.py
│   ├── read_samples_haomo.py
│   ├── read_samples.py
│   └── utils
│       ├── gen_depth_data.py
│       ├── split_train_val.py
│       └── utils.py
├── train
│   ├── training_overlap_transformer_haomo.py
│   └── training_overlap_transformer_kitti.py
├── valid
│   └── valid_seq.py
├── visualize
│   ├── des_list.npy
│   └── viz_haomo.py
└── weights
    ├── pretrained_overlap_transformer_haomo.pth.tar
    └── pretrained_overlap_transformer.pth.tar
```

<!---
To use our code, you need to download the following necessary files and put them in the right positions of the structure above:
- [pretrained_overlap_transformer.pth.tar](https://drive.google.com/file/d/1FNrx9pcDa9NF7z8CFtuTWyauNkeSEFW4/view?usp=sharing): Our pretrained OT on KITTI sequences for easier evaluation.
- [des_list.npy](https://drive.google.com/file/d/13btLQiUokuSHYx229WxtcHGw49-oxmX2/view?usp=sharing): descriptors of Haomo dataset generated by our pretrained OT for visualization.
-->



### Dataset Structure
In the file [config.yaml](https://github.com/haomo-ai/OverlapTransformer/blob/master/config/config.yml), the parameters of `data_root` are described as follows:
```
  data_root_folder (KITTI sequences root) follows:
  ├── 00
  │   ├── depth_map
  │     ├── 000000.png
  │     ├── 000001.png
  │     ├── 000002.png
  │     ├── ...
  │   └── overlaps
  │     ├── train_set.npz
  ├── 01
  ├── 02
  ├── ...
  ├── 10
  └── loop_gt_seq00_0.3overlap_inactive.npz
  
  valid_scan_folder (KITTI sequence 02 velodyne) contains:
  ├── 000000.bin
  ├── 000001.bin
  ...

  gt_valid_folder (KITTI sequence 02 computed overlaps) contains:
  ├── 02
  │   ├── overlap_0.npy
  │   ├── overlap_10.npy
  ...
```
You need to download or generate the following files and put them in the right positions of the structure above:
- You can find the groud truth for KITTI 00  here: [loop_gt_seq00_0.3overlap_inactive.npz](https://drive.google.com/file/d/1upAwJBF-_UIB7R8evW0PuJBM3RnrTbzl/view?usp=sharing)
- You can find `gt_valid_folder` for sequence 02 [here](https://drive.google.com/file/d/13_1j20Uq3ppjVEkYaYcKjiJ2Zm7tudyH/view?usp=sharing).   
- Since the whole KITTI sequences need a large memory, we recommend you generate range images such as `00/depth_map/000000.png` by the preprocessing from [Overlap_Localization](https://github.com/PRBonn/overlap_localization/blob/master/src/prepare_training/gen_depth_and_normal_map.py) or its [C++ version](https://github.com/PRBonn/overlap_localization/tree/master/src/prepare_training/c_utils), and we will not provide these images. Please note that in OverlapTransformer, the `.png` images are used instead of `.npy` files saved in [Overlap_Localization](https://github.com/PRBonn/overlap_localization/blob/master/src/prepare_training/gen_depth_and_normal_map.py).
- More directly, you can generate `.png` range images by [the script from OverlapNet](https://github.com/haomo-ai/OverlapTransformer/blob/master/tools/utils/gen_depth_data.py) updated by us.
- `overlaps` folder of each sequence below `data_root_folder` is provided by the authors of OverlapNet [here](https://drive.google.com/file/d/1i333NUC1DnJglXasqkGYCmo9p45Fx28-/view?usp=sharing). You should rename them to `train_set.npz`.


### Quick Use

For a quick use, you could download our [model pretrained on KITTI](https://drive.google.com/file/d/1FNrx9pcDa9NF7z8CFtuTWyauNkeSEFW4/view?usp=sharing), and the following two files also should be downloaded :
- [calib_file](https://drive.google.com/file/d/1LAcFrRSZQPxdD4EKSwIC0d3-uGvLB3yk/view?usp=sharing): calibration file from KITTI 00.
- [poses_file](https://drive.google.com/file/d/1n02m1OqxK122ce8Cjz_N68PkazGqzj9l/view?usp=sharing): pose file from KITTI 00.

Then you should modify `demo1_config` in the file [config.yaml](https://github.com/haomo-ai/OverlapTransformer/blob/master/config/config.yml).  

Run the demo by:  

```
cd demo
python ./demo_compute_overlap_sim.py
```
You can see a query scan (000000.bin of KITTI 00) with a reprojected positive sample (000005.bin of KITTI 00) and a reprojected negative sample (000015.bin of KITTI 00), and the corresponding similarity.  

<img src="https://github.com/haomo-ai/OverlapTransformer/blob/master/demo.png" width="100%" height="100%">  

Fig. 3 Demo for calculating overlap and similarity with our approach.    


### Training

In the file [config.yaml](https://github.com/haomo-ai/OverlapTransformer/blob/master/config/config.yml), `training_seqs` are set for the KITTI sequences used for training.  

You can start the training with

```
cd train
python ./training_overlap_transformer_kitti.py
```
You can resume from our pretrained model [here](https://github.com/haomo-ai/OverlapTransformer/blob/86cd4a53e1be7029de445ec8b2f3d0fbdb8d38c4/train/training_overlap_transformer_kitti.py#L53) for training.


### Testing

Once a model has been trained , the performance of the network can be evaluated. Before testing, the parameters shoud be set in [config.yaml](https://github.com/haomo-ai/OverlapTransformer/blob/master/config/config.yml)

- `test_seqs`: sequence number for evaluation which is "00" in our work.
- `test_weights`: path of the pretrained model.
- `gt_file`: path of the ground truth file provided by the author of OverlapNet, which can be downloaded [here](https://drive.google.com/file/d/1upAwJBF-_UIB7R8evW0PuJBM3RnrTbzl/view?usp=sharing).


Therefore you can start the testing scripts as follows:

```
cd test
mkdir test_results_kitti
python test_kitti00_prepare.py
python test_kitti00_PR.py
python test_kitti00_topN.py
```
After you run `test_kitti00_prepare.py`, a file named `predicted_des_L2_dis.npz` is generated in `test_results_kitti`, which is used by `python test_kitti00_PR.py` to calculate PR curve and F1max, and used by `python test_kitti00_topN.py` to calculate topN recall.   

For a quick test of the training and testing procedures, you could use our [pretrained model](https://drive.google.com/file/d/1FNrx9pcDa9NF7z8CFtuTWyauNkeSEFW4/view?usp=sharing).  

### Visualization 

#### Visualize evaluation on KITTI 00

Firstly, to visualize evaluation on KITTI 00 with search space, the follwoing three files should be downloaded:
- [calib_file](https://drive.google.com/file/d/1LAcFrRSZQPxdD4EKSwIC0d3-uGvLB3yk/view?usp=sharing): calibration file from KITTI 00.
- [poses_file](https://drive.google.com/file/d/1n02m1OqxK122ce8Cjz_N68PkazGqzj9l/view?usp=sharing): pose file from KITTI 00.
- [cov_file](https://drive.google.com/file/d/1ZaY_OJegIsI0rD5WGOzJ296dh7c7wyiz/view?usp=sharing): covariance file from SUMA++ on KITTI 00.

  
and modify the paths in the file [config.yaml](https://github.com/haomo-ai/OverlapTransformer/blob/master/config/config.yml). Then

```
cd visualize
python viz_kitti.py
```

<img src="https://github.com/haomo-ai/OverlapTransformer/blob/master/query_database_kitti.gif" width="70%" height="70%">  
  
Fig. 4 Evaluation on KITTI 00 with search space from [SuMa++](https://github.com/PRBonn/semantic_suma) (a semantic LiDAR SLAM method).    

#### Visualize evaluation on Haomo challenge 1 (after Haomo dataset is released)

We also provide a visualization demo for Haomo dataset after Haomo dataset is released (Fig. 1). Please download the [descriptors](https://drive.google.com/file/d/13btLQiUokuSHYx229WxtcHGw49-oxmX2/view?usp=sharing) of database (sequence 1-1 of Haomo dataset) firstly and then:  

```
cd visualize
python viz_haomo.py
```

### C++ Implementation

We provide a C++ implementation of OverlapTransformer with libtorch for faster retrival.  
* Please download [.pt](https://drive.google.com/file/d/1oC9_Iyts4r1itu5N3_GAfbdPoZ54C1q4/view?usp=sharing) and put it in the OT_libtorch folder.
* Before building, make sure that [PCL](https://github.com/PointCloudLibrary/pcl) exists in your environment.
* Here we use [LibTorch for CUDA 11.3 (Pre-cxx11 ABI)](https://download.pytorch.org/libtorch/cu113/libtorch-shared-with-deps-1.11.0%2Bcu113.zip). Please modify the path of **Torch_DIR** in [CMakeLists.txt](https://github.com/haomo-ai/OverlapTransformer/blob/master/OT_libtorch/ws/CMakeLists.txt). 
* For more details of LibTorch installation , please check this [website](https://pytorch.org/get-started/locally/).  
Then you can generate a descriptor of 000000.bin of KITTI 00 by
```
cd OT_libtorch/ws
mkdir build
cd build/
cmake ..
make -j6
./fast_ot 
```
You can find our C++ OT can generate a decriptor with **less than 2 ms per frame**.


## Datasets Used by OT

In this section, we list the files of different datasets used by OT for faster inquiry.  

### KITTI Dataset

KITTI is used to validate the place recognition performance in our paper. Currently we have released all the necessary files for evaluation on KITTI. 

* Pretrained model: [pretrained_overlap_transformer.pth.tar](https://drive.google.com/file/d/1FNrx9pcDa9NF7z8CFtuTWyauNkeSEFW4/view)
* Overlaps folder of each sequence for training: [train_set_from_overlapnet.zip](https://drive.google.com/file/d/1i333NUC1DnJglXasqkGYCmo9p45Fx28-/view?usp=sharing).
* Validation folder from sequence 02: [computed_overlap_02.zip](https://drive.google.com/file/d/13_1j20Uq3ppjVEkYaYcKjiJ2Zm7tudyH/view?usp=sharing).   
* The groud truth for sequence 00 for testing: [loop_gt_seq00_0.3overlap_inactive.npz](https://drive.google.com/file/d/1upAwJBF-_UIB7R8evW0PuJBM3RnrTbzl/view?usp=sharing) (You can follow this [issue](https://github.com/PRBonn/OverlapNet/issues/35) to generate this file yourself.)
* Calibration file from the orginal benchmark (00): [calib.txt](https://drive.google.com/file/d/1LAcFrRSZQPxdD4EKSwIC0d3-uGvLB3yk/view?usp=sharing)
* Pose file the orginal benchmark (00): [00.txt](https://drive.google.com/file/d/1n02m1OqxK122ce8Cjz_N68PkazGqzj9l/view?usp=sharing)
* Covariance file from SUMA++ (00): [covariance_2nd.txt](https://drive.google.com/file/d/1ZaY_OJegIsI0rD5WGOzJ296dh7c7wyiz/view?usp=sharing)

### Ford Campus Dataset

Ford is used to validate the generalization ability with zero-shot transferring in our paper. Currently we have released all the necessary preprocessed files of Ford except the code for the evaluation which is similar to KITTI. You just need to follow our existing scripts.

* The overlap-based groud truth for sequence 00 for testing: [loop_gt_seq00_0.3overlap_inactive.npz](https://drive.google.com/file/d/1oYxux_8nrm51foFU7gjv_UWHCQeklKA8/view?usp=sharing)
* The distance-based groud truth for sequence 00 for testing: [loop_gt_seq00_10distance_inactive.npz](https://drive.google.com/file/d/1wSmsK_4PNOR_Omp5Yo9dAb_MfLAxKynX/view?usp=sharing)
* Calibration file from the orginal benchmark (00): [calib.txt](https://drive.google.com/file/d/1yEtzAYcInO-H3QUsQWuQjmmMLKpmujyZ/view?usp=sharing)
* Pose file the orginal benchmark (00): [poses.txt](https://drive.google.com/file/d/1YenoUNTt5e_vRCQN1XJ952A8CfA_eNQs/view?usp=sharing)
* Covariance file from SUMA++ (00): [covariance_2nd.txt](https://drive.google.com/file/d/1OCQhv6rzbdorrl04YSRrGZ_cVLPoGJsf/view?usp=sharing)

### Haomo Dataset

You can find the detailed description of Haomo dataset [here](https://github.com/haomo-ai/OverlapTransformer/tree/master/Haomo_Dataset).

## Related Work

You can find our more recent LiDAR place recognition approaches below, which have better performance on larger time gaps.

* [SeqOT](https://github.com/BIT-MJY/SeqOT): spatial-temporal network using sequential LiDAR data
* [CVTNet](https://github.com/BIT-MJY/CVTNet): cross-view Transformer network using RIVs and BEVs

## License
Copyright 2022, Junyi Ma, Xieyuanli Chen, Jun Zhang, HAOMO.AI Technology Co., Ltd., China.

This project is free software made available under the GPL v3.0 License. For details see the LICENSE file.
