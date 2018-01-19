# This script generates all the metadata necessary to use the ACT-Detector on the AVA Dataset
# This code is not optimized

import pickle
import csv
import numpy as np
import cv2

ava_data = {}

def find_extension_in_file(vid_id, file = "videos/filelist.csv"):
    with open(file) as vidfile:
        vidCSV = csv.reader(vidfile, delimiter=',')
        for vid in vidCSV:
            if(vid[0] == vid_id):
                return(vid[3])

def find_resolution_in_file(vid_id, file = "videos/filelist.csv"):
    with open(file) as vidfile:
        vidCSV = csv.reader(vidfile, delimiter=',')
        for vid in vidCSV:
            if(vid[0] == vid_id):
                return(vid[2])

train_list = "random2_balanced_30_ava_train_v0.6.csv"
test_list = "random2_balanced_30_ava_test_v1.0.csv"

# Add train_videos to list
with open(train_list) as csvfile:
    train_list = []
    already_in_list = []
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        if row[0]+"/"+row[1] not in already_in_list:
            ext_v = find_extension_in_file(row[0])
            train_list.append(row[0]+"/"+str(int(row[1]))+"."+ext_v)
            already_in_list.append(row[0]+"/"+row[1])
    ava_data['train_videos'] = [train_list]

# Add test_videos to list
# NOT FRAMES
with open(test_list) as csvfile:
    test_list = []
    already_in_list = []
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        if row[0]+"/"+row[1] not in already_in_list:
            ext_v = find_extension_in_file(row[0])
            test_list.append(row[0]+"/"+str(int(row[1]))+"."+ext_v)
            already_in_list.append(row[0]+"/"+row[1])
    ava_data['test_videos'] = [test_list]

# Labels
labels = []
label_dict = {}
# parse action names
with open("ava_balanced_list_v1.0.pbtxt") as f:
    while True:
        line1 = f.readline()
        if line1 == '': break # EOF

        line2 = f.readline()
        line3 = f.readline()
        line4 = f.readline()
        line5 = f.readline()

        label_name = ':'.join(line2.split(':')[1:]).strip()[1:-1]
        label_id = line3.split(':')[1].strip()
        label_type = line4.split(':')[1].strip()
        
        labels.append(label_name)
    ava_data['labels'] = labels

# Resolution
res_dict = {}
all_vids = []
for i in range(len(train_list)):
    all_vids.append(train_list[i])
for i in range(len(test_list)):
    all_vids.append(test_list[i])
for vid in all_vids:
    vid_id = vid.split('/')[0]
    res = find_resolution_in_file(vid_id)
    res = res.split('x')
    w = res[0]
    h = res[1]
    res_dict[vid] = (int(h),int(w))
ava_data['resolution'] = res_dict

# nframes
frames_dict = {}
all_vids = []
for i in range(len(train_list)):
    all_vids.append(train_list[i])
for i in range(len(test_list)):
    all_vids.append(test_list[i])
for vid in all_vids:
    cap = cv2.VideoCapture("../AVA_Dataset/frames/clips/"+vid)
    frames_dict[vid] = int(cap.get(7)-1)
    cap.release()
ava_data['nframes'] = frames_dict

with open(train_list) as trainfile, open(test_list) as testfile:
    gttubes_dict = {}
    labels = []
    vid_arrays = []
    trainCSV = csv.reader(trainfile, delimiter=',')
    trainCSV = sorted(trainCSV, key=lambda row: int(row[1]), reverse=False)
    trainCSV = sorted(trainCSV, key=lambda row: row[0], reverse=False)
    previous_id = ""
    previous_url = ""
    for vid in trainCSV:
        if(previous_id != vid[1] and previous_id != ""):
            tubes = {}
            unique_labels = np.unique(labels)
            for i in range(len(unique_labels)):
                y = [vid_arrays[j] for j in range(len(labels)) if labels[j] == unique_labels[i]]
                tubes[int(unique_labels[i])-1]=y
            ext_v = find_extension_in_file(previous_url)
            #ttubes_dict[previous_url+"/"+str(int(previous_id))+"."+ext_v] = {int(labels[0])-1: [vid_arrays[0]]}
            gttubes_dict[previous_url+"/"+str(int(previous_id))+"."+ext_v] = tubes
            labels = []
            vid_arrays = []
        res = find_resolution_in_file(vid[0])
        res = res.split('x')
        w = res[0]
        h = res[1]
        ext_v = find_extension_in_file(vid[0])
        x = [[i, np.rint(float(w)*float(vid[2])), np.rint(float(h)*float(vid[3])), np.rint(float(w)*float(vid[4])), np.rint(float(h)*float(vid[5]))] for i in range(1,frames_dict[vid[0]+"/"+str(int(vid[1]))+"."+ext_v]+1)]

        labels.append(vid[6])
        vid_arrays.append(np.array(x))
        
        previous_url = vid[0]
        previous_id = vid[1]

    testCSV = csv.reader(testfile, delimiter=',')
    testCSV = sorted(testCSV, key=lambda row: int(row[1]), reverse=False)
    testCSV = sorted(testCSV, key=lambda row: row[0], reverse=False)
    for vid in testCSV:
        if(previous_id != vid[1]):
            tubes = {}
            unique_labels = np.unique(labels)
            for i in range(len(unique_labels)):
                y = [vid_arrays[j] for j in range(len(labels)) if labels[j] == unique_labels[i]]
                tubes[int(unique_labels[i])-1]=y
            ext_v = find_extension_in_file(previous_url)
            #ttubes_dict[previous_url+"/"+str(int(previous_id))+"."+ext_v] = {int(labels[0])-1: [vid_arrays[0]]}
            gttubes_dict[previous_url+"/"+str(int(previous_id))+"."+ext_v] = tubes
            labels = []
            vid_arrays = []
        res = find_resolution_in_file(vid[0])
        res = res.split('x')
        w = res[0]
        h = res[1]
        ext_v = find_extension_in_file(vid[0])
        x = [[i, np.rint(float(w)*float(vid[2])), np.rint(float(h)*float(vid[3])), np.rint(float(w)*float(vid[4])), np.rint(float(h)*float(vid[5]))] for i in range(1,frames_dict[vid[0]+"/"+str(int(vid[1]))+"."+ext_v]+1)]

        labels.append(vid[6])
        vid_arrays.append(np.array(x))
        
        previous_url = vid[0]
        previous_id = vid[1]
    tubes = {}
    unique_labels = np.unique(labels)
    for i in range(len(unique_labels)):
        y = [vid_arrays[j] for j in range(len(labels)) if labels[j] == unique_labels[i]]
        tubes[int(unique_labels[i])-1]=y
    ext_v = find_extension_in_file(previous_url)
    #ttubes_dict[previous_url+"/"+str(int(previous_id))+"."+ext_v] = {int(labels[0])-1: [vid_arrays[0]]}
    gttubes_dict[previous_url+"/"+str(int(previous_id))+"."+ext_v] = tubes
    labels = []
    vid_arrays = []
ava_data['gttubes'] = gttubes_dict

print(ava_data.keys())
print(len(ava_data['gttubes']))
print(len(ava_data['train_videos'][0]))
print(len(ava_data['labels']))
print(len(ava_data['test_videos'][0]))
print(len(ava_data['resolution']))
print(len(ava_data['nframes']))

pickle.dump(ava_data, open("AVA-GT.pkl", "wb"), protocol=2)