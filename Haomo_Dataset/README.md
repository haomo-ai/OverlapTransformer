# Haomo Dataset

## Introduction
We provide such a new challenging dataset called **Haomo dataset** to support future research. The dataset was collected by a mobile robot built by [HAOMO.AI Technology company](https://github.com/haomo-ai) equipped with a HESAI PandarXT 32-beam LiDAR sensor in urban environments of Beijing. There are currently five sequences in Haomo dataset as listed in Tab. I. Sequences 1-1 and 1-2 are collected from the same route in 8th December 2021 with opposite driving direction. An additional sequence 1-3 from the same route is utilized as the online query with respect to both 1-1 and 1-2 respectively to evaluate place recognition performance of forward and reverse driving.  

Sequences 2-1 and 2-2 are collected along a much longer route from the same direction, but on different dates, 2-1 on 28th December 2021 and 2-2 on 13th January 2022, where the old one is used as a database while the newer one is used as query. The two sequences are for evaluating the performance for large-scale long-term place recognition.

This dataset will be frequently updated to include highly dynamic and constantly updated traffic scenes. More different types of sensor data are also planned to be included.

TABLE I: Statistics of Haomo dataset
Sequence | 1-1 | 1-2 | 1-3 | 2-1 | 2-2 |
:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
Date | 2021.12.08 | 2021.12.08 | 2021.12.08 | 2021.12.28 | 2022.01.13 |
N_scans | 12500 | 22345 | 13500 | coming | coming |
Length | 2.3 km | 2.3 km | 2.3 km | coming | coming |
Direction | Same | Reverse | - | Same | - |
Role | Database | Database | Query | Database | Query |

## News

* [2022-4-26] Seq. 1-1, 1-2 and 1-3 of Haomo dataset are available.


<img src="https://github.com/haomo-ai/OverlapTransformer/blob/master/Haomo_Dataset/haomo_dataset.png" width="90%"/>  

<!---
<img src="https://github.com/haomo-ai/OverlapTransformer/blob/master/Haomo_Dataset/dataset_short_term.gif" width="40%"/>
-->



## Download

You can download the scans and poses of the LiDAR sensor from the following links.

#### Sequence 1-1 and 1-2

* [[scans](https://perception-data.oss-cn-beijing.aliyuncs.com/loc/place_recognition/OT/1-1and1-2/scans.zip)] (11.3 G)  
* [[poses](https://perception-data.oss-cn-beijing.aliyuncs.com/loc/place_recognition/OT/1-1and1-2/1-1and1-2.txt)] (5.3 M)  

#### Sequence 1-3

* [[scans](https://perception-data.oss-cn-beijing.aliyuncs.com/loc/place_recognition/OT/1-3/scans.zip)] (12 G)  
* [[poses](https://perception-data.oss-cn-beijing.aliyuncs.com/loc/place_recognition/OT/1-3/1-3.txt)] (5.7 M)  
* [[transformation between the first poses](https://perception-data.oss-cn-beijing.aliyuncs.com/loc/place_recognition/OT/1-3/transformation_bet_traj.txt)] (405 B)

#### Other sequences

Coming soon ...

## Format

* **[scans]** contains all .bin files of recorded point clouds from the 32-beam LiDAR.
* **[poses]** contains the local poses of the LiDAR sensor.
* **[transformation between the first poses]** contains the transformation matrix between the first poses of two trajectories. (e.g., transformation matrix of Seq 1-3 is T<sub>1-3-0</sub><sup>-1</sup>·T<sub>1-1-0</sub>)

Our dataset follows the data format of [KITTI odometry benchmark](http://www.cvlibs.net/datasets/kitti/eval_odometry.php). 

## Publication
If you use our Haomo dataset in your academic work, please cite the corresponding paper ([PDF](https://arxiv.org/pdf/2203.03397.pdf)):  
    
	@article{ma2022arxiv, 
		author = {Junyi Ma and Jun Zhang and Jintao Xu and Rui Ai and Weihao Gu and Cyrill Stachniss and Xieyuanli Chen},
		title  = {{OverlapTransformer: An Efficient and Rotation-Invariant Transformer Network for LiDAR-Based Place Recognition}},
		journal = {arXiv preprint},
		eprint = {2203.03397},
		year = {2022}
	}

