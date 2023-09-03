
import time, pdb, argparse, subprocess
from os import listdir, path
from glob import glob
from SyncNetInstance import *
from tqdm import tqdm

# ==================== LOAD PARAMS ====================


parser = argparse.ArgumentParser(description = "SyncNet")

parser.add_argument('--initial_model', type=str, default="data/syncnet_v2.model", help='')
parser.add_argument('--videofile', type=str, default="data/example.avi", help='')
parser.add_argument('--maxoffset', type=int, default=3, help='')
parser.add_argument('--minoffset', type=int, default=-3, help='')
parser.add_argument('--data_root', type=str, help='', required=True)
parser.add_argument('--batch_size', type=int, default='20', help='')
parser.add_argument('--vshift', type=int, default='15', help='')
parser.add_argument('--tmp_dir', type=str, default="data/work/pytmp", help='')
parser.add_argument('--reference', type=str, default="demo", help='')

args = parser.parse_args()


# ==================== RUN EVALUATION ====================

s = SyncNetInstance()

s.loadParameters(args.initial_model)
print("Model %s loaded."%args.initial_model)
faces = os.listdir(args.data_root)
index = 0
with tqdm(total=200) as pbar:
    with open('offsets.txt', 'w') as f:
        for face in faces:
            videos = os.listdir(os.path.join(args.data_root, face))
            for video in videos:
                try:
                    offset, conf, dists = s.evaluate(args, videofile=os.path.join(args.data_root, face, video))
                    if offset > 3 or offset < -3 or conf < 3:
                        f.write('File {}  -- AVoffset is {} --confidence is {} \n'.format(os.path.join(args.data_root, face, video), offset, conf))
                        os.remove(os.path.join(args.data_root, face, video))
                        pbar.update(1)
                    else:
                        pbar.update(1)
                except:
                    f.write('File {}  -- wrong data \n'.format(os.path.join(args.data_root, face, video)))
                    os.remove(os.path.join(args.data_root, face, video))
                    pbar.update(1)

def rename_videos(data_root):
    faces = os.listdir(data_root)
    for face in faces:
        videos = os.listdir(os.path.join(data_root, face))
        for index, video in enumerate(videos):
            iii = str(index).zfill(5)
            os.rename(os.path.join(data_root, face, video), os.path.join(data_root, face, iii+'.mp4'))
rename_videos(args.data_root)



