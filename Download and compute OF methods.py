import sys
import os

import csv
import pafy

import numpy as np
from PIL import Image
import cv2
#sys.path.insert(0, '/work/croig/matlab-py/install/lib/python2.7/site-packages')
#import matlab.engine

# Method for downloading a list of videos from youtube, having a csv with a column of ids as input
# The outputs are a directory with all the videos and a file with the information about them
def download_dataset_youtube(file_list, out_dir, csv_position = 0):
	# ---------------- Inputs -------------------
	# file_list:	CSV file with the links in a column
	# out_dir:		Directory where the videos will be stored
	# csv_position:	Column position of the CSV

	# ---------------- Outputs -------------------
	# Videos stored in outdir
	# File list in outdir/filelist.csv with video id, duration, resolution, extension

	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	if not os.path.isfile(out_dir+"/filelist.csv"):
		file = open(out_dir+"/filelist.csv", 'w')
		file.close()
	counter_id = 0
	already_in_set = False
	errors = []
	with open(file_list) as csvfile:
		readCSV = csv.reader(csvfile, delimiter=',')
		for row in readCSV:
			saved_vids = open(out_dir+"/filelist.csv", 'r+')
			svidsCSV_reader = csv.reader(saved_vids, delimiter=',')
			# Check if we already downloaded the video
			for line in svidsCSV_reader:
				already_in_set = False
				if line[0] == row[0] or row[0] in errors:
					already_in_set=True
					break
			if already_in_set:
				continue

			# The video is not in the dataset, start downloading it
			url = "https://www.youtube.com/watch?v="+row[0]
			print(url)
			try:
				video = pafy.new(url)
			except(IOError, UnicodeEncodeError) as e:
				errors.append(row[0])
				continue
			best = video.getbest()
			filename = best.download(filepath=out_dir+"/"+row[0]+"."+best.extension)

			# Save the video infromation in the saved vids file
			saved_vids.write(row[0]+","+video.duration+","+best.resolution+","+best.extension+"\n")
			already_in_set = False
			saved_vids.close()

# Crashes Memory Error, probably VideoCapture not closed
def extract_OF_Brox(file_list, vid_list, out_dir, csv_position = 0):
	eng = matlab.engine.start_matlab()
	create_flow_path = "./"
	eng.addpath(create_flow_path)
	clips_path = "../AVA_Dataset/frames/clips_test/"
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	with open(file_list) as csvfile:
		clipsCSV = csv.reader(csvfile, delimiter=',')
		for clips_row in clipsCSV:
			with open(vid_list) as csvvid:
				folderCSV = csv.reader(csvvid, delimiter=',')
				for folder_row in folderCSV:
					if(clips_row[0] == folder_row[0]):
						if not os.path.exists(out_dir+"/"+clips_row[0]):
							os.makedirs(out_dir+"/"+clips_row[0])
						vid_id = clips_row[1]
						if not os.path.exists(out_dir+"/"+clips_row[0]+"/"+vid_id):
							os.makedirs(out_dir+"/"+clips_row[0]+"/"+vid_id)
						else:
							continue
						print(vid_id)
						#print(clips_path+folder_row[0]+"/"+str(int(vid_id))+"."+folder_row[3])
						cap = cv2.VideoCapture(clips_path+folder_row[0]+"/"+str(int(vid_id))+"."+folder_row[3])
						number_of_frames=int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
						#print("Number of frames:",number_of_frames)
						#Go frame by frame
						past_image = Image.fromarray(cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB))
						frame_size = past_image.size
						for i in range(1,number_of_frames):
							im = cap.read()
							if im[0]:
								current_image = Image.fromarray(cv2.cvtColor(im[1], cv2.COLOR_BGR2RGB))
								past_image_mat = matlab.uint8(list(past_image.getdata()))
								past_image_mat.reshape((past_image.size[0], past_image.size[1], 3))	
								current_image_mat = matlab.uint8(list(current_image.getdata()))
								current_image_mat.reshape((current_image.size[0], current_image.size[1], 3))	
								#Compute flow
								save_path = out_dir+"/"+clips_row[0]+"/"+vid_id+"/flow{:0>3}.jpg".format(i-1)
								eng.create_flow(past_image_mat, current_image_mat, save_path,nargout=0)
								past_image = current_image
							else:
								save_path = out_dir+"/"+clips_row[0]+"/"+vid_id+"/flow{:0>3}.jpg".format(i)
								eng.create_flow(past_image_mat, current_image_mat, save_path,nargout=0)
						#Call matlab and extract OF
						#eng.create_flow(past_image_mat, current_image, out_dir+"/"+clips_row[0]+"/"+vid_id+"/flow{:0>3}.jpg".format(i-1))

# Crashes Memory Error, probably same error
def extract_OF_Brox_batch_18(file_list, vid_list, out_dir, worker_i, csv_position = 0):
	eng = matlab.engine.start_matlab()
	create_flow_path = "./"
	eng.addpath(create_flow_path)
	clips_path = "../AVA_Dataset/frames/clips_test/"
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	with open(vid_list) as csvvid:
		folderCSV = csv.reader(csvvid, delimiter=',')
		i=0	
		for folder_row in folderCSV:		
			if(i%18 == worker_i):
				with open(file_list) as csvfile:
					clipsCSV = csv.reader(csvfile, delimiter=',')
					for clips_row in clipsCSV:
						if(clips_row[0] == folder_row[0]):
							if not os.path.exists(out_dir+"/"+clips_row[0]):
								os.makedirs(out_dir+"/"+clips_row[0])
							vid_id = clips_row[1]
							if not os.path.exists(out_dir+"/"+clips_row[0]+"/"+vid_id):
								os.makedirs(out_dir+"/"+clips_row[0]+"/"+vid_id)
							else:
								continue
							print(vid_id)
							#print(clips_path+folder_row[0]+"/"+str(int(vid_id))+"."+folder_row[3])
							cap = cv2.VideoCapture(clips_path+folder_row[0]+"/"+str(int(vid_id))+"."+folder_row[3])
							number_of_frames=int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
							#print("Number of frames:",number_of_frames)
							#Go frame by frame
							past_image = Image.fromarray(cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB))
							frame_size = past_image.size
							for i in range(1,number_of_frames):
								im = cap.read()
								if im[0]:
									current_image = Image.fromarray(cv2.cvtColor(im[1], cv2.COLOR_BGR2RGB))
									past_image_mat = matlab.uint8(list(past_image.getdata()))
									past_image_mat.reshape((past_image.size[0], past_image.size[1], 3))	
									current_image_mat = matlab.uint8(list(current_image.getdata()))
									current_image_mat.reshape((current_image.size[0], current_image.size[1], 3))	
									#Compute flow
									save_path = out_dir+"/"+clips_row[0]+"/"+vid_id+"/flow{:0>3}.jpg".format(i-1)
									eng.create_flow(past_image_mat, current_image_mat, save_path,nargout=0)
									past_image = current_image
								else:
									save_path = out_dir+"/"+clips_row[0]+"/"+vid_id+"/flow{:0>3}.jpg".format(i)
									eng.create_flow(past_image_mat, current_image_mat, save_path,nargout=0)
							#Call matlab and extract OF
							#eng.create_flow(past_image_mat, current_image, out_dir+"/"+clips_row[0]+"/"+vid_id+"/flow{:0>3}.jpg".format(i-1))
			i+=1

#def extract_denseflow_tvl1():

def extract_OF_Farneback_sbatch(file_list, vid_list, out_dir, worker_i, num_workers, csv_position = 0):
	eng = matlab.engine.start_matlab()
	create_flow_path = "./"
	eng.addpath(create_flow_path)
	clips_path = "../AVA_Dataset/frames/clips/"
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	with open(vid_list) as csvvid:
		folderCSV = csv.reader(csvvid, delimiter=',')
		w=0	
		for folder_row in folderCSV:		
			if(w%num_workers == worker_i):
				with open(file_list) as csvfile:
					clipsCSV = csv.reader(csvfile, delimiter=',')
					for clips_row in clipsCSV:
						if(clips_row[0] == folder_row[0]):
							if not os.path.exists(out_dir+"/"+clips_row[0]):
								os.makedirs(out_dir+"/"+clips_row[0])
							vid_id = clips_row[1]
							if not os.path.exists(out_dir+"/"+clips_row[0]+"/"+vid_id):
								os.makedirs(out_dir+"/"+clips_row[0]+"/"+vid_id)
							else:
								continue
							#print(vid_id)
							#print(clips_path+folder_row[0]+"/"+str(int(vid_id))+"."+folder_row[3])
							cap = cv2.VideoCapture(clips_path+folder_row[0]+"/"+str(int(vid_id))+"."+folder_row[3])
							#number_of_frames=int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
							#print("Number of frames:",number_of_frames)
							#Go frame by frame
							success, frame1 = cap.read()
							#print(frame1)
							prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
							#print(prvs)
							hsv = np.zeros_like(frame1)
							i=1
							while(success):
								success,im = cap.read()
								if success:
									next = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
									#print(prvs)
									#print(hsv)
									#Compute flow Farneback
									flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
									scale = 16
									mag = np.sqrt(np.power(flow[...,0],2)+np.power(flow[...,1],2))*scale+128;
									mag = np.minimum(mag, 255)
									flow = flow*scale+128
									flow = np.minimum(flow,255)
									flow = np.maximum(flow,0)

									hsv[...,2] = flow[...,0]
									hsv[...,1] = flow[...,1]
									hsv[...,0] = mag
									save_path = out_dir+"/"+clips_row[0]+"/"+vid_id+"/{:0>5}.jpg".format(i)
									cv2.imwrite(save_path,hsv)
									prvs = next
								else:
									save_path = out_dir+"/"+clips_row[0]+"/"+vid_id+"/{:0>5}.jpg".format(i)
									cv2.imwrite(save_path,hsv)
								i+=1
							#Call matlab and extract OF
							#eng.create_flow(past_image_mat, current_image, out_dir+"/"+clips_row[0]+"/"+vid_id+"/flow{:0>3}.jpg".format(i-1))
			w+=1

def generate_directories_from_clips_folder(directory,save_path):
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	for root,dirs,files in os.walk(directory):
		#print(root)
		for file in files:
				#print(file.split(".")[1])
				if(file.split(".")[1] in ["webm","mp4","avi","mkv"]):
					save_clip = save_path+"/"+root.split("/")[6]+"/"+file.split(".")[0]
					#print(save_clip)
					if not os.path.exists(save_clip):
						os.makedirs(save_clip)

def extract_frames_from_directory(directory,save_path, worker_i, num_workers):
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	i = 0
	for root,dirs,files in os.walk(directory):
		#print(root)
		for file in files:
			if(i%num_workers == worker_i):
				#print(file.split(".")[1])
				if(file.split(".")[1] in ["webm","mp4", "avi","mkv"]):
					#print(root+"/"+file)
					vidcap = cv2.VideoCapture(root+"/"+file)
					success,image = vidcap.read()
					count = 1
					success = True
					save_clip = save_path+"/"+root.split("/")[6]+"/"+file.split(".")[0]
					print(save_clip)
					if not os.path.exists(save_clip):
						os.makedirs(save_clip)
					while success:
						success,image = vidcap.read()
						#print('Read a new frame: ', success)
						if(success):
							cv2.imwrite(save_clip+"/"+"{:0>5}.jpg".format(count), image)  
							count += 1
			i+=1

# Delete lines with repeated information in csv_position
# NOT GENERIC FUNCTION
def delete_repeated_lines(file_in, file_out, csv_position=1):
	if not os.path.isfile(file_out):
		file = open(file_out, 'w')
		file.close()
	with open(file_in) as csvfile, open(file_out, 'w') as savefile:
		lines = csv.reader(csvfile, delimiter=',')
		aux_line = 0
		for line in lines:
			if(line[csv_position] != aux_line):
				savefile.write(line[0]+","+line[csv_position]+"\n")
				aux_line = line[csv_position]


if __name__=="__main__":
	exec(sys.argv[1])