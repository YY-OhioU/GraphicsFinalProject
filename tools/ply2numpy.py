import pickle
from pathlib import Path

import numpy as np
from plyfile import PlyData

CWD = Path(__file__).parent
PROJECT_ROOT = CWD.parent

plydata = PlyData.read(PROJECT_ROOT / "models" / "point_cloud.ply")
print(plydata)
print(plydata["vertex"][0])
converted = np.array(plydata['vertex']).astype(
    [('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('red', 'f4'), ('green', 'f4'), ('blue', 'f4')])
converted['red'] /= 255.0
converted['green'] /= 255.0
converted['blue'] /= 255.0
print(type(plydata['vertex'][0]))
first_row = converted[0]
print("converted: ", first_row)

pickle.dump(converted, open(PROJECT_ROOT / 'models' / 'point_cloud.pkl', 'wb'))
