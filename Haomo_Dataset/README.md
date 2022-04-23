# Haomo Dataset


We provide such a new challenging dataset called Haomo dataset to support future research. The dataset was collected by a mobile robot built by [HAOMO.AI Technology company](https://github.com/haomo-ai) equipped with a HESAI PandarXT 32-beam LiDAR sensor in urban environments of Beijing. There are currently five sequences in Haomo dataset as listed in Tab. I. Sequences 1-1 and 1-2 are collected from the same route in 8th December 2021 with opposite driving direction. An additional sequence 1-3 from the same route is utilized as the online query with respect to both 1-1 and 1-2 respectively to evaluate place recognition performance of forward and reverse driving.  

Sequences 2-1 and 2-2 are collected along a much longer route from the same direction, but on different dates, 2-1 on 28th December 2021 and 2-2 on 13th January 2022, where the old one is used as a database while the newer one is used as query. The two sequences are for evaluating the performance for large-scale long-term place recognition.

This dataset will be frequently updated to include highly dynamic and constantly updated traffic scenes. More different types of sensor data are also planned to be included.

TABLE I: Statistics of Haomo dataset
Sequence | 1-1 | 1-2 | 1-3 | 2-1 | 2-2 |
:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
Date | 2021.12.08 | 2021.12.08 | 2021.12.08 | 2021.12.28 | 2022.01.13 |
N_scans | 12500 | 22345 | 13500 | 100887 | 88154 |
Length | 2.3 km | 2.3 km | 2.3 km | 11.5 km | 11.5 km |
Direction | Same | Reverse | - | Same | - |
Role | Database | Database | Query | Database | Query |

The download link and the corresponding SDK of Haomo dataset are coming soon ...
