from moviepy.editor import VideoFileClip
input_video_path = "/mnt/minio/node77/liuzheng/RJ/Data/mpg/RJ_mpg/u6797 2021_03_26/u6797 2021_03_26.mpg"
# 指定输出AVI视频文件的路径
output_video_path = "/mnt/minio/node77/liuzheng/RJ/Data/mpg/RJ_mpg/u6797 2021_03_26/u6797 2021_03_26.avi"
clip = VideoFileClip(input_video_path)
clip.write_videofile(output_video_path, codec='libx264')
clip.close()