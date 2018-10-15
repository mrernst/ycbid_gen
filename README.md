### README for YCBID_gen

#### Prerequisites
To use this piece of software you need to have the following companion programmes installed

* gazebo simulator
* icub model world
* yarp
* numpy
* pandas
* PIL

#### Usage

Download the 3d laserscans with textures from the ycb-webpage > link , put them in /models then run the model converter ./model_converter.sh path/to/tar/model/files/folder.

The database generator then is used in the following way:
python3 ycbid_gen.py --scenes 10 --n_occluders 2 --objects_in_universe 15 --desired_occlusion 0.2