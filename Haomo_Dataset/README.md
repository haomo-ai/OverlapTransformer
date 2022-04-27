# Haomo Dataset

## Introduction
We provide such a new challenging dataset called **Haomo dataset** to support future research. The dataset was collected by a mobile robot built by [HAOMO.AI Technology company](https://github.com/haomo-ai) equipped with a HESAI PandarXT 32-beam LiDAR sensor in urban environments of Beijing. There are currently five sequences in Haomo dataset as listed in Tab. I. Sequences 1-1 and 1-2 are collected from the same route in 8th December 2021 with opposite driving direction. An additional sequence 1-3 from the same route is utilized as the online query with respect to both 1-1 and 1-2 respectively to evaluate place recognition performance of forward and reverse driving.  

Sequences 2-1 and 2-2 are collected along a much longer route from the same direction, but on different dates, 2-1 on 28th December 2021 and 2-2 on 13th January 2022, where the old one is used as a database while the newer one is used as query. The two sequences are for evaluating the performance for large-scale long-term place recognition.

This dataset will be frequently updated to include highly dynamic and constantly updated traffic scenes. More different types of sensor data are also planned to be included.

TABLE I: Statistics of Haomo dataset
Sequence | 1-1 | 1-2 | 1-3 | 2-1 | 2-2 |
:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
Date | 2021.12.08 | 2021.12.08 | 2021.12.08 | 2021.12.28 | 2022.01.13 |
N_scans | 12500 | 22345 | 13500 | 100887 | 88154 |
Length | 2.3 km | 2.3 km | 2.3 km | 11.5 km | 11.1 km |
Direction | Same | Reverse | - | Same | - |
Role | Database | Database | Query | Database | Query |


<img src="https://github.com/haomo-ai/OverlapTransformer/blob/master/Haomo_Dataset/haomo_dataset.png" width="90%"/>  

<img src="https://github.com/haomo-ai/OverlapTransformer/blob/master/Haomo_Dataset/dataset_short_term.gif" width="40%"/>


## Download

You can download the scans and poses of the LiDAR sensor from the following links.

#### Sequence 1-1 and 1-2

* [[scans](https://www.ipb.uni-bonn.de/html/projects/Haomo/1-1and1-2/scans.zip)]  
* [[poses](https://www.ipb.uni-bonn.de/html/projects/Haomo/1-1and1-2/1-1and1-2.txt)]  

#### Sequence 1-3

* [[scans](https://www.ipb.uni-bonn.de/html/projects/Haomo/1-3/scans.zip)]  
* [[poses](https://www.ipb.uni-bonn.de/html/projects/Haomo/1-3/1-3.txt)]  

#### Other sequences

Coming soon ...

## Format

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

