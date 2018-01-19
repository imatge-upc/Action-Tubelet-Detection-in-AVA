import os
import subprocess
import cv2
import svgwrite
import pickle

import pdb

videodir = "videos/"
annotfile = "ava_train_v0.5.csv"
actionlistfile = "ava_action_list_v1.0.pbtxt"
outdir = "frames"

outpath_dset = outdir + "/ava.pkl"
outdir_clips = outdir + "/clips/"
outdir_midframe = outdir + "/midframes/"


clip_length = 3 # seconds
clip_time_padding = 1.0 # seconds

# util
def hou_min_sec(millis):
    millis = int(millis)
    seconds=(millis/1000)%60
    seconds = int(seconds)
    minutes=(millis/(1000*60))%60
    minutes = int(minutes)
    hours=(millis/(1000*60*60))
    return ("%d:%d:%d" % (hours, minutes, seconds))

def _supermakedirs(path, mode):
    if not path or os.path.exists(path):
        return []
    (head, tail) = os.path.split(path)
    res = _supermakedirs(head, mode)
    os.mkdir(path)
    os.chmod(path, mode)
    res += [path]
    return res

def mkdir_p(path):
    try:
        _supermakedirs(path, 0775)
    except OSError as exc: # Python >2.5
        pass


"""
label {
  name: "bend/bow (at the waist)"
  label_id: 1
  label_type: PERSON_MOVEMENT
}
"""
label_dict = {}
# parse action names
with open(actionlistfile) as f:
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
		
		label_dict[label_id] = [label_type, label_name]

# extract all clips and middle frames
with open(annotfile) as f:
	annots = f.read().splitlines()
	

#The format of a row is the following: video_id, middle_frame_timestamp, person_box, action_id
#-5KQ66BBWC4,0904,0.217,0.008,0.982,0.966,12


annot_number = 0

for ann in annots:
	# PARSE ANNOTATION STRING
	
	seg = ann.split(',')
	
	video_id = seg[0]
	middle_frame_timestamp = int(seg[1]) # in seconds
	left_bbox = float(seg[2]) # from 0 to 1
	top_bbox = float(seg[3])
	right_bbox = float(seg[4])
	bottom_bbox = float(seg[5])
	action_id = seg[6]
	
	
	videofile_noext = videodir + '/' + video_id
	videofile = subprocess.check_output('ls %s*' % videofile_noext, shell=True)
	videofile = videofile.split()[0]
	
	video_extension = videofile.split('.')[-1]
	
	# OPEN VIDEO FOR INFORMATION IF NECESSARY
	vcap = cv2.VideoCapture(videofile) # 0=camera
	if vcap.isOpened(): 
		# get vcap property 
		vidwidth = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
		vidheight = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float

		# or
		vidwidth = vcap.get(3)  # float
		vidheight = vcap.get(4) # float
	else:
		exit(1)
	
	
	# EXTRACT KEYFRAME WITH FFMPEG
	
	print middle_frame_timestamp
	
	outdir_keyframe = outdir_midframe + '/' + video_id + '/'
	mkdir_p(outdir_keyframe)
	
	outpath = outdir_keyframe + '%d.jpg' % (int(middle_frame_timestamp))
	ffmpeg_command = 'rm %(outpath)s; ffmpeg -ss %(timestamp)f -i %(videopath)s -frames:v 1 %(outpath)s' % {
		'timestamp': float(middle_frame_timestamp),
		'videopath': videofile,
		'outpath': outpath
		}
	
	subprocess.call(ffmpeg_command, shell=True)
	#subprocess.call('eog %(outpath)s &' % {'outpath': outpath}, shell=True)
	
	
	# EXTRACT CLIPS WITH FFMPEG
	
	print middle_frame_timestamp
	
	outdir_clip = outdir_clips + '/' + video_id + '/'
	mkdir_p(outdir_clip)
	
	# ffmpeg -i a.mp4 -force_key_frames 00:00:09,00:00:12 out.mp4
	# ffmpeg -ss 00:00:09 -i out.mp4 -t 00:00:03 -vcodec copy -acodec copy -y final.mp4
	
	clip_start = middle_frame_timestamp - clip_time_padding - float(clip_length)/2
	if clip_start < 0:
		clip_start = 0
	clip_end = middle_frame_timestamp + float(clip_length)/2
	
	outpath_clip = outdir_clip + '%d.%s' % (int(middle_frame_timestamp), video_extension)
	#outpath_clip_tmp = outpath + '_tmp.%s' % video_extension
	
	ffmpeg_command = 'rm %(outpath)s; ffmpeg -ss %(start_timestamp)s -i %(videopath)s -g 1 -force_key_frames 0 -t %(clip_length)d %(outpath)s' % {
		'start_timestamp': hou_min_sec(clip_start*1000),
		'end_timestamp': hou_min_sec(clip_end*1000),
		'clip_length': clip_length + clip_time_padding,
		'videopath': videofile,
		'outpath': outpath_clip
		}
	
	subprocess.call(ffmpeg_command, shell=True)
	#subprocess.call('eog %(outpath)s &' % {'outpath': outpath}, shell=True)
	
	
	
	
	
	# LOAD GENERATED JPG, CREATE SVG, OVERLAY BOUNDING BOX
	outdir_keyframe_svg = outdir + '/visu_svg/' + video_id + '/'
	mkdir_p(outdir_keyframe_svg)
	
	outpath_svg = outdir_keyframe_svg + '%d_%s_%d.svg' % (int(middle_frame_timestamp), action_id, annot_number )

	svg_document = svgwrite.Drawing(filename = outpath_svg,
	                                size = ("%dpx" % vidwidth, "%dpx" % vidheight))
	
	svg_document.add(svg_document.image(outpath))
	
	#    ADD BBOX
	insert_left = vidwidth * left_bbox
	insert_top = vidheight * top_bbox
	bbox_width = vidwidth * (right_bbox - left_bbox)
	bbox_height = vidheight * (bottom_bbox - top_bbox)
	
	svg_document.add(svg_document.rect(insert = (insert_left, insert_top),
		size = ("%dpx" % bbox_width, "%dpx" % bbox_height),
		stroke_width = "5",
		stroke = "rgb(255,100,100)",
		fill = "rgb(255,100,100)",
		fill_opacity=0.0)
	)
	
	#    ADD ACTION CAPTION
	actioncaption = ''
	label_type_and_name = label_dict.get(action_id)
	if label_type_and_name != None:
		svg_document.add(svg_document.text("%s" % (label_type_and_name[0]),
			insert = (10, 20),
			fill = "rgb(255,30,30)",
			font_size = "22px")
		)
		svg_document.add(svg_document.text("%s" % (label_type_and_name[1]),
			insert = (10, 40),
			fill = "rgb(255,30,30)",
			font_size = "22px",
			font_weight = "bold")
		)
	
	print svg_document.tostring()
	svg_document.save()
	
	outdir_visu_jpg = outdir + '/visu_jpg/' + video_id + '/'
	mkdir_p(outdir_visu_jpg)
	outpath_visu_jpg = outdir_visu_jpg + '%d_%s_%d.jpg' % (int(middle_frame_timestamp), action_id, annot_number )
	
	subprocess.call('convert -- "%s" "%s"' % (outpath_svg, outpath_visu_jpg), shell=True)
	
	
	print ""
	print "ffmpeg_command:", ffmpeg_command
	print "middle_frame_timestamp:", middle_frame_timestamp
	print "time:", hou_min_sec(middle_frame_timestamp)
	print "action_id:", action_id
	
	annot_number += 1
	
	#pdb.set_trace()



