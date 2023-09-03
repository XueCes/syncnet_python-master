from sklearn.model_selection import train_test_split
from pydub import AudioSegment
import cv2
import ffmpeg
import math
import tqdm
import argparse
import os
import subprocess
import shutil

# import demo_syncnet as syncnet

duration = 4
FPS = 25
data_root = 'F:\\AIGC\\datasetPartCN6AVI'
input_dir = 'F:\\AIGC\\datasetPartCN6AVI'  # 输入文件夹路径
output_dir = 'F:\\AIGC\\datasetPartCN1'  # 输出文件夹路径

# parser = argparse.ArgumentParser(description='Code to train the Wav2Lip model WITH the visual quality discriminator')
#
# parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset", required=True, type=str)
#
# parser.add_argument('--target_root', help='Save checkpoints to this directory', required=True, type=str)
# parser.add_argument('--duration', help='Load the pre-trained Expert discriminator', required=True, type=str)
#
# args = parser.parse_args()

def convert_video2Fps(src):
    # cmd = f"ffmpeg -i {src} -r 25 25fps.mp4 -y"
    cmd = f"ffmpeg -y -r 25 -i {src} 25fps.mp4"
    os.system(cmd)
    print(f'remove src: {src}')
    os.rename(src, 'temp.mp4')
    try:
        shutil.move('25fps.mp4', f"{src}")
        os.remove('temp.mp4')
    except:
        os.rename('temp.mp4', src)
        print(f'file {src} wrong processing')


# TODO 确认FPS
def check_fps(filename):
    cap = cv2.VideoCapture(filename)
    fps = cap.get(cv2.CAP_PROP_FPS)
    return fps


# TODO FPS转换
def convert_videos2fps(data_root):
    fps = FPS
    identities = os.listdir(data_root)
    identities = [x if not x.endswith('.DS_Store') else os.remove(os.path.join(data_root, x)) for x in identities]
    identities.sort()
    for identity in identities:
        # if int(identity) < 6:
        #     continue
        fullPathID = os.path.join(data_root, identity)
        videos = os.listdir(fullPathID)
        videos = [x if x.endswith('.mp4') else os.remove(os.path.join(data_root, identity, x)) for x in videos]
        i = 0
        for video in videos:
            src = os.path.join(fullPathID, video)
            if check_fps(src) == fps:
                continue
            print(f'Convert video: {src} to 25 fps')
            convert_video2Fps(src)


# TODO(构建训练集、测试集、验证集txt文件)
def get_dataset_txt(data_root):
    lines = []
    players = os.listdir(data_root)
    for item in players:
        videos = os.listdir(os.path.join(data_root, item))
        for vv in videos:
            lines.append(str(item) + '/' + vv)
    train_txt, test_txt = train_test_split(lines, test_size=0.2, train_size=0.8, shuffle=True)
    test_txt, val_txt = train_test_split(test_txt, test_size=0.5, train_size=0.5, shuffle=True)
    with open('filelists/train.txt', 'wb') as f:
        for item in train_txt:
            f.write((item + '\n').encode())
    with open('filelists/test.txt', 'wb') as f:
        for item in test_txt:
            f.write((item + '\n').encode())
    with open('filelists/val.txt', 'wb') as f:
        for item in val_txt:
            f.write((item + '\n').encode())
    return train_txt, test_txt, val_txt


'''
# TODO(检查预处理后音频和视频时常是否一致)
def check_sync(data_root):
    "检测预处理后的音视频是否同步"
    with open('check_sample_log.txt', 'w') as fp:
        for item in os.listdir(data_root):
            for processed_vedio in os.listdir(os.path.join(data_root, item)):
                files = os.listdir(os.path.join(data_root, item, processed_vedio))
                if '.jpg' in [end[-4:] for end in files]:
                    # 判断是否音视频同步
                    # 先判断音频时长
                    audio = AudioSegment.from_file(os.path.join(data_root, item, processed_vedio, 'audio.wav'))
                    audio_duration = len(audio)
                    video_duration = ((len(files) - 1) / 25) * 1000
                    dif = abs(audio_duration - video_duration)
                    # if dif > 100 and dif < 200:
                    #     fp.write("样本 {} 音视差为 {} ms \n".format(os.path.join(data_root,item,processed_vedio), dif))
                    if dif != 0:
                        fp.write("样本 {} 音视差为 {} ms , 误差严重，需要确认！！！\n".format(
                            os.path.join(data_root, item, processed_vedio), dif))
                        # shutil.move(os.path.join(data_root, item, processed_vedio),
                        #             os.path.join("dataCheckedError", item))
                    else:
                        fp.write("样本 {} 音视差为 {} ms , 没有误差\n".format(
                            os.path.join(data_root, item, processed_vedio), dif))
                else:
                    fp.write("样本 {} 为错误样本 \n".format(os.path.join(data_root, item, processed_vedio)))
'''

# TODO（切分视频）
def split_video(input_file, output_dir, output_prefix, duration):
    probe = ffmpeg.probe(input_file)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    duration_sec = math.ceil(float(video_info['duration']))
    num_segments = math.ceil(duration_sec / duration)

    for i in range(num_segments):
        start_time = i * duration
        end_time = min((i + 1) * duration, duration_sec)
        output_file = os.path.join(output_dir, f"{output_prefix}_{i}.mp4")
        ffmpeg.input(input_file, ss=start_time, t=end_time - start_time).output(output_file, crf=1).run()

def split_video_by_dir(data_root, output_dir, t):
    faces = os.listdir(data_root)
    os.mkdir(output_dir)
    for face in faces:
        os.mkdir(os.path.join(output_dir, face))
        videos = os.listdir(os.path.join(data_root, face))
        for index, video in enumerate(videos):
            split_video(os.path.join(data_root, face, video), os.path.join(output_dir, face), str(index), t)


# TODO（删除时长小于2s的视频）
def clean_videos(data_root):
    faces = os.listdir(data_root)
    for face in faces:
        videos = os.listdir(os.path.join(data_root, face))
        for index, video in enumerate(videos):
            cap = cv2.VideoCapture(os.path.join(data_root, face, video))
            if cap.isOpened():
                rate = cap.get(5)
                frame_num = cap.get(7)
                duration = frame_num / rate
                cap.release()
                if duration < 2:
                    os.remove(os.path.join(data_root, face, video))
                else:
                    continue


# TODO（视频文件改名）
def rename_videos(data_root):
    faces = os.listdir(data_root)
    for face in faces:
        videos = os.listdir(os.path.join(data_root, face))
        for index, video in enumerate(videos):
            iii = str(index).zfill(5)
            os.rename(os.path.join(data_root, face, video), os.path.join(data_root, face, iii+'.mp4'))


# 所有视频转码


# 递归遍历目录并创建对应的子目录

# 递归遍历目录
def traverse_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file_name in files:
            if file_name.endswith('.avi'):
                input_file = os.path.join(root, file_name)
                output_file = os.path.join(output_dir, os.path.relpath(input_file, input_dir).replace('.avi', '.mp4'))

                # 使用FFmpeg进行转码
                command = ['ffmpeg', '-i', input_file, '-c:v', 'libx264', '-crf', '1', '-preset', 'medium', '-c:a', 'aac', '-b:a', '128k', output_file]
                subprocess.run(command)
                os.remove(input_file)


traverse_directory(data_root)


# 视频帧率转换
convert_videos2fps(data_root)

# 切分视频
split_video_by_dir(data_root, output_dir, duration)
# 删除时长小于2秒的视频
clean_videos(output_dir)
# 视频文件改名
# rename_videos(output_dir)
# 检查音视频同步

# 构建训练集、测试集、验证集txt
# train_txt, test_txt, val_txt = get_dataset_txt(output_dir)

# split_video_by_dir(args.data_root, args.target_root, duration)
# clean_videos(args.target_root)
# rename_videos(args.target_root)


