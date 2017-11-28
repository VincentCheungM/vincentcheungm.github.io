---
title: "Ground Plane Fitting"
date: 2017-11-25
tags: ["pcl", "ros", "python"]
draft: false
---

#Ground Plane Fitting

[repo](https://github.com/VincentCheungM/Run_based_segmentation)

```
@inproceedings{Zermas2017Fast,
  title={Fast segmentation of 3D point clouds: A paradigm on LiDAR data for autonomous vehicle applications},
  author={Zermas, Dimitris and Izzat, Izzat and Papanikolopoulos, Nikolaos},
  booktitle={IEEE International Conference on Robotics and Automation},
  year={2017},
}
```

In this paper, Zermas proposed a scan_line_based point cloud segmentation method. The ground plane is removed according to `Ground Plan Fitting`. Here, the ground plane model as $a*x+b*y+c*z + d = 0$. And the `normal` of plane is regared as the least primary component of the covariance matrix.

And here is the Python implentation of GPF. Numpy, PCL and ROS is required. **The following code used 5 point_field including the `ring`, if you use `KITTI` dataset, the point_field should be set to 4.**

```python
import numpy as np
from numpy import linalg as la 

import os
import argparse

import rospy
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2

class GPF(object):
    def __init__(self,n_segs=3, n_iter=3, n_lpr=20, th_seeds=0.4, th_dist=0.2,
        lis_topic='/velodyne_points', pub_topic='ground_topic', field_num=5):
        self.n_segs = n_segs
        self.n_iter = n_iter
        self.n_lpr = n_lpr
        self.th_seeds = th_seeds
        self.th_dist = th_dist

        ## ros related
        self.lis_topic = lis_topic
        self.pub_topic = pub_topic
        self.point_field = self.make_point_field(field_num)

        # publisher
        self.pcmsg_pub1 = rospy.Publisher(self.pub_topic, PointCloud2,queue_size=1)
        self.pcmsg_pub2 = rospy.Publisher('no'+self.pub_topic, PointCloud2,queue_size=1)
        # ros node init
        rospy.init_node('gpf_node', anonymous=True)
        # listener
        rospy.Subscriber(self.lis_topic, PointCloud2, self.pcmsg_cb)
    
    def ExtractInitialSeeds(self, point):
        """
        args:
        `point`:shape[n,5],[x,y,z,intensity,ring]
        """
        p_sort = point[np.lexsort(point[:,:3].T)][:self.n_lpr]
        lpr = np.mean(p_sort[:,2])
        cond = point[:,2] <(lpr+self.th_seeds)
        return point[cond]
    
    def main(self, seeds, point):
        pg = seeds
        cov = np.cov(pg[:,:3].T)
        for i in range(self.n_iter):
            # estimate plane
            cov = np.cov(pg[:,:3].T)
            s_mean = np.mean(pg[:,:3],axis=0)
            U,sigma,VT=la.svd(cov)
            normal = U[:,-1]
            d = -np.dot(normal.T,s_mean)
            # condition
            th=self.th_dist - d
            cond_pg = np.dot(normal,point[:,:3].T)<th
            
            pg = point[cond_pg]
            png = point[~cond_pg]
        return pg,png

    def make_point_field(self, num_field):
        # get from data.fields
        msg_pf1 = pc2.PointField()
        msg_pf1.name = np.str('x')
        msg_pf1.offset = np.uint32(0)
        msg_pf1.datatype = np.uint8(7)
        msg_pf1.count = np.uint32(1)

        msg_pf2 = pc2.PointField()
        msg_pf2.name = np.str('y')
        msg_pf2.offset = np.uint32(4)
        msg_pf2.datatype = np.uint8(7)
        msg_pf2.count = np.uint32(1)

        msg_pf3 = pc2.PointField()
        msg_pf3.name = np.str('z')
        msg_pf3.offset = np.uint32(8)
        msg_pf3.datatype = np.uint8(7)
        msg_pf3.count = np.uint32(1)

        msg_pf4 = pc2.PointField()
        msg_pf4.name = np.str('intensity')
        msg_pf4.offset = np.uint32(16)
        msg_pf4.datatype = np.uint8(7)
        msg_pf4.count = np.uint32(1)

        if num_field == 4:
            return [msg_pf1, msg_pf2, msg_pf3, msg_pf4]
        
        msg_pf5 = pc2.PointField()
        msg_pf5.name = np.str('ring')
        msg_pf5.offset = np.uint32(20)
        msg_pf5.datatype = np.uint8(4)
        msg_pf5.count = np.uint32(1)

        return [msg_pf1, msg_pf2, msg_pf3, msg_pf4, msg_pf5]

    def pcmsg_cb(self,data):
        # read points from pointcloud message `data`
        pc = pc2.read_points(data, skip_nans=True, field_names=("x", "y", "z","intensity","ring"))
        # to conver pc into numpy.ndarray format
        np_p = np.array(list(pc))
        np_p = np_p[np_p[:,2]>-4]
        seeds = self.ExtractInitialSeeds(np_p)
        pg, png = self.main(seeds,np_p)
        print pg.shape, png.shape
        new_header = data.header
        #new_header.frame_id = '/velodyne'
        new_header.stamp = rospy.Time()

        pc_ranged_pg = pc2.create_cloud(header=new_header,fields=self.point_field,points=pg)
        pc_ranged_png = pc2.create_cloud(header=new_header,fields=self.point_field,points=png)
        
        self.pcmsg_pub1.publish(pc_ranged_pg)
        self.pcmsg_pub2.publish(pc_ranged_png)

if __name__ =='__main__':
    n_segs = 3
    n_iter = 2
    n_lpr = 20
    th_seeds = 1.2#0.4
    th_dist = 0.3#0.5
    gpf = GPF(n_segs,n_iter,n_lpr,th_seeds,th_dist)
    rospy.spin()
    # Below is some sample code for saving PCD for display.
    # pcd=np.loadtxt('./test_pcd/99_1504941053170163000.pcd',skiprows=11)

    # seeds = gpf.ExtractInitialSeeds(pcd)

    # pg, png = gpf.main(seeds,pcd)
    # print pg.shape, png.shape
    # #display
    # pg[:,3]=50
    # png[:,3]=255
    # header1=('# .PCD v.7 - Point Cloud Data file format\n'
    # 'VERSION .7\n'
    # 'FIELDS x y z intensity ring\n'
    # 'SIZE 4 4 4 4 4\n'
    # 'TYPE F F F F U\n'
    # 'COUNT 1 1 1 1 1\n'
    # 'WIDTH '+str(pg.shape[0])+'\n'
    # 'HEIGHT 1\n'
    # 'VIEWPOINT 0 0 0 1 0 0 0\n'
    # 'POINTS '+str(pg.shape[0])+'\n'
    # 'DATA bin\n')
    # header2=('# .PCD v.7 - Point Cloud Data file format\n'
    # 'VERSION .7\n'
    # 'FIELDS x y z intensity ring\n'
    # 'SIZE 4 4 4 4 4\n'
    # 'TYPE F F F F U\n'
    # 'COUNT 1 1 1 1 1\n'
    # 'WIDTH '+str(png.shape[0])+'\n'
    # 'HEIGHT 1\n'
    # 'VIEWPOINT 0 0 0 1 0 0 0\n'
    # 'POINTS '+str(png.shape[0])+'\n'
    # 'DATA bin\n')
    # #print header
    # np.savetxt('./'+'ground'+'.pcd',pg,fmt='%f %f %f %f %u',header=header1,comments='')
    # np.savetxt('./'+'notground'+'.pcd',png,fmt='%f %f %f %f %u',header=header2,comments='')

```