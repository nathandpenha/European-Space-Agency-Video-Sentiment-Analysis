import ffmpeg
import os

from os import listdir

from os.path import isfile, join

from os import walk

from os import walk

input_directory = 'D:\\Projects\\esA\\us10\\Dataset\\prototype\\'  # This directory should exist with individual actor directories in it. eg ./vidoes/actor1/video.mp4

input_path = input_directory

videos = []  # Will contain all the list of videos to be converted.

actor_directory = []  # Will be used to keep track of which actor directroies are present in './videos' directory.

frame_per_second = 6  # set the number of frames required per second here.

print('Frames Per Second is set to: ' + str(frame_per_second))

print('Images will be stored in the ./results directory')

print('Actors Directories Found in ./videos are:')

for actor in os.listdir(input_directory):

    if os.path.isdir(input_directory + actor):

        print(actor)

        for video in os.listdir(input_directory + actor):
            actor_directory.append('./results/' + actor + '/' + video + '/')

            videos.append(input_directory + actor + '/' + video)

try:

    # creating a folder named data if it does not already exist

    for dirs in actor_directory:

        if not os.path.exists(dirs):
            os.makedirs(dirs)

        if not os.path.exists(dirs):
            os.makedirs(dirs)





# if not created then raise error

except OSError:

    print('Error: Creating directory of data')

counter = -1;  # used to keep track of which video is being processed.

for video in videos:

    counter = counter + 1

    print('Creating frames in --' + actor_directory[counter])

    try:

        (ffmpeg.input(video)

         .filter('fps', fps=frame_per_second)

         .output(actor_directory[counter] + '/frame' + '%d.jpg',

                 # video_bitrate='5000k',     #Uncomment based on requirement

                 # s='64x64',                         #Uncomment based on requirement

                 # sws_flags='bilinear',      #Uncomment based on requirement

                 start_number=0)

         .run(capture_stdout=True, capture_stderr=True))

    except ffmpeg.Error as e:

        print('stdout:', e.stdout.decode('utf8'))

        print('stderr:', e.stderr.decode('utf8'))