''' 将视频分为4s/段 '''
import subprocess

def getVideoTime(path):
    cmdline = 'ffprobe "%s" -show_entries format=duration -of compact=p=0:nk=1 -v 0'%path
    gettime=subprocess.check_output(cmdline, shell=True)
    timeT=int(float(gettime.strip()))
    return timeT

videoPath = 'F:\\AIGC\\vedio\\2\\2.mp4'
cutTime = 4
timeT = getVideoTime(videoPath)
firstTime = 0
index = 1
while firstTime < timeT:
    cmdLine = 'ffmpeg -ss %s -i %s -c copy -t %s %s.mp4 -loglevel quiet -y' % (firstTime, videoPath, cutTime, '%s' % index)
    print(cmdLine)
    returnCmd = subprocess.call(cmdLine, shell=True)
    firstTime += cutTime
    index += 1
