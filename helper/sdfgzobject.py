import sys
sys.path.insert(0, "../src/")
import RESSOURCES
import gazebo as gz
import numpy as np
import time


class SDFGzObject(gz.GzObject):
    def __init__(self, world_controler, name, object_path, x, y, z, roll=0, pitch=0, yaw=0):
        super().__init__(world_controler, name)
        self.world_controler.load_model_from_file(object_path)
        time.sleep(0.5)
        self.set_positions(x, y, z, roll, pitch, yaw)
