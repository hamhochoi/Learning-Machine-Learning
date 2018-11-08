from read_csv import find_ad_list
import os


videofile = "/home/hamhochoi/Videos/TV/07.mp4"
csv_file = "/media/hamhochoi/Beo/OneDrive for Business 1/OneDrive - student.hust.edu.vn/OD/VCCorp/PySceneDetect/PySceneDetect-0.4/myvideo_scenes_06.csv"
# csv_file = "/home/hamhochoi/Documents/PySceneDetect/PySceneDetect-0.4/myscene.csv"
outputfile = "output1.mp4"

ad_list = find_ad_list(csv_file)
print (ad_list)

start_time = ad_list[0]

for i in range(1, len(ad_list)):
	if (ad_list[i] == ad_list[i-1]):
		end_time = ad_list[i]

# start_time = "0"
# end_time   = "15:00"

# os.system("ffmpeg -i " + videofile + " -acodec copy -vcodec copy -ss " + str(start_time) + " -t " + str(end_time) + " " + outputfile)