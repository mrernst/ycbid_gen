### README for YCBID_gen

#### Prerequisites
To use this piece of software you need to have the following companion programmes installed

* gazebo simulator [http://gazebosim.org](http://gazebosim.org)
* icub model world [icub gazebo worlds](https://github.com/robotology/icub-gazebo)
* yarp [http://www.yarp.it](http://www.yarp.it)
* numpy
* pandas
* PIL

#### Usage

Download the tar compressed 16k 3d laserscans with textures from the ycb webpage [Link](http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com), put them in /models then run the model converter 

./model_converter.sh path/to/tar/model/files/folder.

The database generator then is used in the following way:
python3 ycbid_gen.py --scenes 10 --n_occluders 2 --objects_in_universe 15 --desired_occlusion 0.2