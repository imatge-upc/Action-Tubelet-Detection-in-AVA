import sys
import os
import cPickle as pickle

import cv2
import numpy as np

CAFFE_PYTHON_PATH = os.path.join(os.path.dirname(__file__), "../python")
sys.path.insert(0, CAFFE_PYTHON_PATH)
import caffe
from Dataset import GetDataset
from ACT_utils import *

K = 6
IMGSIZE = 300
MEAN = np.array([[[104, 117, 123]]], dtype=np.float32)
NFLOWS = 5

def extract_tubelets(dname, gpu=-1, redo=False):
    """Extract the tubelets for a given dataset

    args:
        - dname: dataset name (example: 'JHMDB')
        - gpu (default -1): use gpu given in argument, or use cpu if -1
        - redo: wheter or not to recompute already computed files

    save a pickle file for each frame
    the file contains a tuple (dets, dets_all)
        - dets is a numpy array with 2+4*K columns containing the tubelets starting at this frame after per-class nms at 0.45 and thresholding the scores at 0.01
          the columns are <label> <score> and then <x1> <y1> <x2> <y2> for each of the frame in the tubelet
        - dets_all contains the tubelets obtained after a global nms at 0.7 and thresholding the scores at 0.01
            it is a numpy arrray with 4*K + L + 1 containing the coordinates of the tubelets and the scores for all labels

    note: this version is inefficient: it is better to estimate the per-frame features once
    """
    d = GetDataset(dname)

    if gpu >= 0:
        caffe.set_mode_gpu()
        caffe.set_device(gpu)

    model_dir = os.path.join(os.path.dirname(__file__), '../models/ACT-detector/', dname)
    output_dir = os.path.join(os.path.dirname(__file__), '../results/ACT-detector/', dname)
    
    # load the RGB network
    rgb_proto = os.path.join(model_dir, "deploy_RGB.prototxt")
    rgb_model = os.path.join(model_dir, "../generated_AVA_iter_118662.caffemodel")
    net_rgb = caffe.Net(rgb_proto, caffe.TEST, weights=rgb_model)
    
    # load the FLOW5 network
    flo_proto = os.path.join(model_dir, "deploy_FLOW5.prototxt")
    flo_model = os.path.join(model_dir, "../generated_AVA_iter_59463.caffemodel")
    net_flo = caffe.Net(flo_proto, caffe.TEST, weights=flo_model)

    vlist = d.test_vlist()
    for iv, v in enumerate(vlist):
        print("Processing video {:d}/{:d}: {:s}".format( iv+1, len(vlist), v))
        h, w = d.resolution(v)
        
        # network output is normalized between 0,1 ; so we will multiply it by the following array
        resolution_array = np.array([w,h,w,h]*K, dtype=np.float32)
        
        # now process each frame
        for i in xrange(1, 1 + d.nframes(v) - K + 1):
            outfile = os.path.join(output_dir, d.frame_format(v,i) + ".pkl")
            
            # skip if already computed
            if os.path.isfile(outfile) and not redo:
                continue
            
            # read the frames for the forward
            kwargs_rgb  = {}
            kwargs_flo = {}
            for j in xrange(K):
                cap = cv2.VideoCapture(d.vidfile(v,0))
                #print(frame)
                #print(int(cap.get(7)))
                cap.set(1,i + j - 1)
                im = cap.read()[1]
                cap.release()
                #im = cv2.imread(d.imfile(v, i + j))
                if im is None:
                    print "Image {:s} does not exist".format(d.imfile(v, i+j))
                    return
                imscale = cv2.resize(im, (IMGSIZE, IMGSIZE), interpolation=cv2.INTER_LINEAR)
                kwargs_rgb['data_stream' + str(j)] = np.transpose(imscale-MEAN, (2, 0, 1))[None, :, :, :]
                imf = [cv2.imread(d.flowfile(v.split(".")[0], min(d.nframes(v), i + j + iflow))) for iflow in xrange(NFLOWS)]
                if np.any(imf) is None:
                    print "Flow image {:s} does not exist".format(d.flowfile(v, i+j))
                    return
                imscalef = [cv2.resize(im, (IMGSIZE, IMGSIZE), interpolation=cv2.INTER_LINEAR) for im in imf]
                timscale = [np.transpose(im-MEAN, (2, 0, 1))[None, :, :, :] for im in imscalef]
                kwargs_flo['data_stream' + str(j) + 'flow'] = np.concatenate(timscale, axis=1)
            
            # compute rgb and flow scores
            # two forward passes: one for the rgb and one for the flow 
            net_rgb.forward(end="mbox_conf_flatten", **kwargs_rgb) # forward of rgb with confidence and regression
            net_flo.forward(end="mbox_conf_flatten", **kwargs_flo) # forward of flow5 with confidence and regression
            
            # compute late fusion of rgb and flow scores (keep regression from rgb)
            # use net_rgb for standard detections, net_flo for having all boxes
            scores = 0.5 * (net_rgb.blobs['mbox_conf_flatten'].data + net_flo.blobs['mbox_conf_flatten'].data)
            net_rgb.blobs['mbox_conf_flatten'].data[...] = scores
            net_flo.blobs['mbox_conf_flatten'].data[...] = scores
            net_flo.blobs['mbox_loc'].data[...] = net_rgb.blobs['mbox_loc'].data
            
            # two forward passes, only for the last layer 
            # dets is the detections after per-class NMS and thresholding (stardard)
            # dets_all contains all the scores and regressions for all tubelets 
            dets = net_rgb.forward(start='detection_out')['detection_out'][0, 0, :, 1:]
            dets_all = net_flo.forward(start='detection_out_full')['detection_out_full'][0, 0, :, 1:]
            
            # parse detections with per-class NMS
            if dets.shape[0] == 1 and np.all(dets == -1):
                dets = np.empty((0, dets.shape[1]), dtype=np.float32)

            dets[:, 2:] *= resolution_array # network output was normalized in [0..1]
            dets[:, 0] -= 1 # label 0 was background, come back to label in [0..nlabels-1]
            dets[:, 2::2] = np.maximum(0, np.minimum(w, dets[:, 2::2]))
            dets[:, 3::2] = np.maximum(0, np.minimum(h, dets[:, 3::2]))

            # parse detections with global NMS at 0.7 (top 300)
            # coordinates were normalized in [0..1]
            dets_all[:, 0:4*K] *= resolution_array 
            dets_all[:, 0:4*K:2] = np.maximum(0, np.minimum(w, dets_all[:, 0:4*K:2]))
            dets_all[:, 1:4*K:2] = np.maximum(0, np.minimum(h, dets_all[:, 1:4*K:2]))
            idx = nms_tubelets(np.concatenate((dets_all[:, :4*K], np.max(dets_all[:, 4*K+1:], axis=1)[:, None]), axis=1), 0.7, 300)
            dets_all = dets_all[idx, :]
            
            # save file
            if not os.path.isdir(os.path.dirname(outfile)):
                os.system('mkdir -p ' + os.path.dirname(outfile))

            with open(outfile, 'wb') as fid:
                pickle.dump((dets, dets_all), fid)


def load_frame_detections(d, vlist, dirname, nms):
    if isinstance(d, str):
        d = GetDataset(d)

    alldets = [] # list of numpy array with <video_index> <frame_index> <ilabel> <score> <x1> <y1> <x2> <y2>
    for iv, v in enumerate(vlist):
        h,w = d.resolution(v)
        
        # aggregate the results for each frame
        vdets = {i: np.empty((0,6), dtype=np.float32) for i in range(1, 1 + d.nframes(v))} # x1, y1, x2, y2, score, ilabel
        
        # load results for each starting frame
        for i in xrange(1, 1 + d.nframes(v) - K + 1):
            resname = os.path.join(dirname, d.frame_format(v,i) + '.pkl')
            
            if not os.path.isfile(resname):
                print("ERROR: Missing extracted tubelets "+resname)
                sys.exit()

            with open(resname, 'rb') as fid:
                dets, _ = pickle.load(fid)
            
            if dets.size == 0:
                continue

            for k in xrange(K):
                vdets[i+k] = np.concatenate( (vdets[i+k],dets[:,np.array([2+4*k,3+4*k,4+4*k,5+4*k,1,0])] ), axis=0)

        # Perform NMS in each frame
        for i in vdets:
            idx = np.empty((0,), dtype=np.int32)
            for ilabel in xrange(d.nlabels):
                a = np.where(vdets[i][:,5] == ilabel)[0]
                
                if a.size == 0:
                    continue
                
                idx = np.concatenate((idx, a[nms2d(vdets[i][vdets[i][:, 5] == ilabel, :5], nms)]), axis=0)
            
            if idx.size == 0:
                continue

            alldets.append(np.concatenate((iv * np.ones((idx.size, 1), dtype=np.float32), i * np.ones((idx.size, 1), dtype=np.float32), vdets[i][idx, :][:, np.array([5, 4, 0, 1, 2, 3], dtype=np.int32)]), axis=1))
    
    return np.concatenate(alldets, axis=0)

def frameAP(dname, th=0.5, redo=False):
    d = GetDataset(dname)
    dirname = os.path.join(os.path.dirname(__file__), '../results/ACT-detector/', dname)
    
    eval_file = os.path.join(dirname, "frameAP{:g}.pkl".format(th))
    
    if os.path.isfile(eval_file) and not redo:
        with open(eval_file, 'rb') as fid:
            res = pickle.load(fid)
    else:
        vlist = d.test_vlist()
        print(vlist)
        # load per-frame detections
        alldets = load_frame_detections(d, vlist, dirname, 0.3)
        res = {}
        
        # compute AP for each class
        for ilabel,label in enumerate(d.labels):
            # detections of this class
            detections = alldets[alldets[:, 2] == ilabel, :]
            
            # load ground-truth of this class
            gt = {}
            for iv, v in enumerate(vlist):
                tubes = d.gttubes(v)
                
                if not ilabel in tubes:
                    continue
                
                for tube in tubes[ilabel]:
                    for i in xrange(tube.shape[0]):
                        k = (iv, int(tube[i, 0]))
                        if not k in gt:
                            gt[k] = []
                        gt[k].append(tube[i, 1:5].tolist())

            for k in gt:
                gt[k] = np.array( gt[k] )
            
            # pr will be an array containing precision-recall values
            pr = np.empty((detections.shape[0] + 1, 2), dtype=np.float32)# precision,recall
            pr[0, 0] = 1.0
            pr[0, 1] = 0.0
            fn = sum([g.shape[0] for g in gt.values()]) # false negatives
            fp = 0 # false positives
            tp = 0 # true positives
            
            for i, j in enumerate(np.argsort(-detections[:,3])):
                k = (int(detections[j,0]), int(detections[j,1]))
                box = detections[j, 4:8]
                ispositive = False
                
                if k in gt:
                    ious = iou2d(gt[k], box)
                    amax = np.argmax(ious)
                    
                    if ious[amax] >= th:
                        ispositive = True
                        gt[k] = np.delete(gt[k], amax, 0)
                        
                        if gt[k].size == 0:
                            del gt[k]
                
                if ispositive:
                    tp += 1
                    fn -= 1
                else:
                    fp += 1
                if((tp+fn)>0):
                    pr[i+1, 0] = float(tp) / float(tp + fp)
                    pr[i+1, 1] = float(tp) / float(tp + fn)
                else:
                    pr[i+1, 0] = 0
                    pr[i+1, 1] = 0
            res[label] = pr
        
        # save results
        with open(eval_file, 'wb') as fid:
            pickle.dump(res, fid)
    
    # display results
    ap = 100*np.array([pr_to_ap(res[label]) for label in d.labels])
    print "frameAP"
    
    for il, _ in enumerate(d.labels):
        print("{:20s} {:8.2f}".format('', ap[il]))
    
    print("{:20s} {:8.2f}".format("mAP", np.mean(ap)))
    print("")


def frameMABO(dname, redo=False):
    d = GetDataset(dname)
    dirname = os.path.join( os.path.dirname(__file__), '../results/ACT-detector/', dname)
    eval_file = os.path.join(dirname, "frameMABO.pkl")
    
    if os.path.isfile(eval_file) and not redo:
        with open(eval_file, 'rb') as fid:
            BO = pickle.load(fid)
    else:
        vlist = d.test_vlist()
        BO = {l: [] for l in d.labels} # best overlap
        
        for v in vlist:
            gt = d.gttubes(v)
            h, w = d.resolution(v)
            
            # load per-frame detections
            vdets = {i: np.empty((0,4), dtype=np.float32) for i in range(1, 1+d.nframes(v))}

            # load results for each chunk
            for i in xrange(1, 1 + d.nframes(v) - K + 1):
                resname = os.path.join(dirname, d.frame_format(v,i) + '.pkl')
                if not os.path.isfile(resname):
                    print("ERROR: Missing extracted tubelets " + resname)
                    sys.exit()
                
                with open(resname, 'rb') as fid:
                    dets, _ = pickle.load(fid)
                
                for k in xrange(K):
                    vdets[i+k] = np.concatenate((vdets[i + k], dets[:, 2+4*k:6+4*k]), axis=0)
            
            # for each frame
            for i in xrange(1, 1 + d.nframes(v)):
                for ilabel in gt:
                    label = d.labels[ilabel]
                    for t in gt[ilabel]:
                        # the gt tube does not cover frame i
                        if not i in t[:,0]:
                            continue

                        gtbox = t[t[:,0] == i, 1:5] # box of gt tube at frame i
                        
                        if vdets[i].size == 0: # we missed it
                            BO[label].append(0)
                            continue

                        ious = iou2d(vdets[i], gtbox)
                        BO[label].append( np.max(ious) )
            # save file
            with open(eval_file, 'wb') as fid:
                pickle.dump( BO, fid)

    # print MABO results
    ABO = {la: 100 * np.mean(np.array(BO[la])) for la in d.labels} # average best overlap
    
    for la in d.labels:
        print("{:20s} {:6.2f}".format(la, ABO[la]))

    print("{:20s} {:6.2f}".format("MABO", np.mean(np.array(ABO.values()))))


def frameCLASSIF(dname, redo=False):
    d = GetDataset(dname)
    dirname = os.path.join(os.path.dirname(__file__), '../results/ACT-detector/', dname)
    eval_file = os.path.join(dirname, "frameCLASSIF.pkl")
    
    if os.path.isfile(eval_file) and not redo:
        with open(eval_file, 'rb') as fid:
            CLASSIF = pickle.load(fid)
    else:
        vlist = d.test_vlist()
        #print(vlist)
        CORRECT = [0 for ilabel in xrange(d.nlabels)]
        TOTAL   = [0 for ilabel in xrange(d.nlabels)]
        
        for v in vlist:
            nframes = d.nframes(v)
            # load all tubelets
            VDets = {}
            for startframe in xrange(1, nframes + 2 - K):
                resname = os.path.join(dirname, d.frame_format(v, startframe) + '.pkl')
                
                if not os.path.isfile(resname):
                    print("ERROR: Missing extracted tubelets " + resname)
                    sys.exit()
                
                with open(resname, 'rb') as fid:
                    _, VDets[startframe] = pickle.load(fid)
            
            # iterate over ground-truth
            tubes = d.gttubes(v)
            for ilabel in tubes:
                for g in tubes[ilabel]:
                    for i in xrange(g.shape[0]):
                        frame = int(g[i, 0])
                        
                        # just in case a tube is longer than the video
                        if frame > nframes:
                            continue

                        gtbox = g[i, 1:5]
                        scores = np.zeros((d.nlabels,), dtype=np.float32)
                        
                        # average the score over the 6 frames
                        for sf in xrange(max(1, frame - K + 1), min(nframes - K + 1, frame) + 1):
                            overlaps = iou2d(VDets[sf][:, 4*(frame-sf):4*(frame-sf)+4], gtbox)
                            scores += np.sum(VDets[sf][overlaps >= 0.7, 4*K + 1:],axis=0)
                        
                        # check classif
                        if np.argmax(scores) == ilabel:
                            CORRECT[ilabel] += 1

                        TOTAL[ilabel] += 1
        print(TOTAL)
        print(CORRECT)
        CLASSIF = [float(CORRECT[ilabel]) / float(TOTAL[ilabel]) for ilabel in xrange(d.nlabels) if TOTAL[ilabel] != 0 ]
        
        with open(eval_file, 'wb') as fid:
            pickle.dump(CLASSIF, fid)

    # print classif results
    for il, la in enumerate(d.labels):
        print("{:20s} {:6.2f}".format(la, 100*CLASSIF[il]))

    print("{:20s} {:6.2f}".format("CLASSIF", 100*np.mean(np.array(CLASSIF))))


def BuildTubes(dname, redo=False):
    d = GetDataset(dname)
    dirname = os.path.join( os.path.dirname(__file__), '../results/ACT-detector/', dname)
    vlist = d.test_vlist()

    for iv, v in enumerate(vlist):
        print("Processing video {:d}/{:d}: {:s}".format(iv + 1, len(vlist), v))

        outfile = os.path.join(dirname, v + "_tubes.pkl")
        
        if os.path.isfile(outfile) and not redo:
            continue
        
        RES = {}
        nframes = d.nframes(v)
        
        # load detected tubelets
        VDets = {}
        for startframe in xrange(1, nframes + 2 - K):
            resname = os.path.join(dirname, d.frame_format(v, startframe) + '.pkl')
            
            if not os.path.isfile(resname):
                print("ERROR: Missing extracted tubelets " + resname)
                sys.exit()
            
            with open(resname, 'rb') as fid:
                _, VDets[startframe] = pickle.load(fid)
        
        for ilabel in xrange(d.nlabels):
            FINISHED_TUBES = []
            CURRENT_TUBES = [] # tubes is a list of tuple (frame, lstubelets)

            def tubescore(tt):
                return np.mean(np.array([tt[i][1][-1] for i in xrange(len(tt))]))

            for frame in xrange(1, d.nframes(v) + 2 - K):
                # load boxes of the new frame and do nms while keeping Nkeep highest scored
                ltubelets = VDets[frame][:,range(4*K) + [4*K + 1 + ilabel]] # Nx(4K+1) with (x1 y1 x2 y2)*K ilabel-score
                idx = nms_tubelets(ltubelets, 0.3, top_k=10)
                ltubelets = ltubelets[idx,:]
                
                # just start new tubes
                if frame == 1:
                    for i in xrange(ltubelets.shape[0]):
                        CURRENT_TUBES.append( [(1,ltubelets[i,:])] )
                    continue

                # sort current tubes according to average score
                avgscore = [tubescore(t) for t in CURRENT_TUBES ]
                argsort = np.argsort(-np.array(avgscore))
                CURRENT_TUBES = [CURRENT_TUBES[i] for i in argsort]
                
                # loop over tubes
                finished = []
                for it, t in enumerate(CURRENT_TUBES):
                    # compute ious between the last box of t and ltubelets
                    last_frame, last_tubelet = t[-1]
                    ious = []
                    offset = frame - last_frame
                    if offset < K:
                        nov = K - offset
                        ious = sum([iou2d(ltubelets[:, 4*iov:4*iov+4], last_tubelet[4*(iov+offset):4*(iov+offset+1)]) for iov in xrange(nov)])/float(nov)
                    else:
                        ious = iou2d(ltubelets[:, :4], last_tubelet[4*K-4:4*K])
                    
                    valid = np.where(ious >= 0.2)[0]
                    
                    if valid.size>0:
                        # take the one with maximum score
                        idx = valid[ np.argmax(ltubelets[valid, -1])]
                        CURRENT_TUBES[it].append((frame, ltubelets[idx,:]))
                        ltubelets = np.delete(ltubelets, idx, axis=0)
                    else:
                        # skip
                        if offset>=5:
                            finished.append(it)

                # finished tubes that are done
                for it in finished[::-1]: # process in reverse order to delete them with the right index
                    FINISHED_TUBES.append( CURRENT_TUBES[it][:])
                    del CURRENT_TUBES[it]
                
                # start new tubes
                for i in xrange(ltubelets.shape[0]):
                    CURRENT_TUBES.append([(frame,ltubelets[i,:])])

            # all tubes are not finished
            FINISHED_TUBES += CURRENT_TUBES

            # build real tubes
            output = []
            for t in FINISHED_TUBES:
                score = tubescore(t)
                
                # just start new tubes
                if score< 0.01:
                    continue
                
                beginframe = t[0][0]
                endframe = t[-1][0]+K-1
                length = endframe+1-beginframe
                
                # delete tubes with short duraton
                if length < 15:
                    continue

                # build final tubes by average the tubelets
                out = np.zeros((length, 6), dtype=np.float32)
                out[:, 0] = np.arange(beginframe,endframe+1)
                n_per_frame = np.zeros((length, 1), dtype=np.int32)
                for i in xrange(len(t)):
                    frame, box = t[i]
                    for k in xrange(K):
                        out[frame-beginframe+k, 1:5] += box[4*k:4*k+4]
                        out[frame-beginframe+k, -1] += box[-1]
                        n_per_frame[frame-beginframe+k ,0] += 1
                out[:,1:] /= n_per_frame
                output.append((out, score))

            RES[ilabel] = output
        
        with open(outfile, 'wb') as fid:
            pickle.dump(RES, fid)

def videoAP(dname, th=0.5, redo=False):
    d = GetDataset(dname)
    dirname = os.path.join( os.path.dirname(__file__), '../results/ACT-detector/', dname)
    eval_file = os.path.join(dirname, "videoAP{:g}.pkl".format(th))
    
    if os.path.isfile(eval_file) and not redo:
        with open(eval_file, 'rb') as fid:
            res = pickle.load(fid)
    else:
        vlist = d.test_vlist()
        
        # load detections
        # alldets = for each label in 1..nlabels, list of tuple (v,score,tube as Kx5 array)
        alldets = {ilabel: [] for ilabel in xrange(d.nlabels)}
        for v in vlist:
            tubename = os.path.join(dirname, v + '_tubes.pkl')
            if not os.path.isfile(tubename):
                print("ERROR: Missing extracted tubes " + tubename)
                sys.exit()

            
            with open(tubename, 'rb') as fid:
                tubes = pickle.load(fid)
            
            for ilabel in xrange(d.nlabels):
                ltubes = tubes[ilabel]
                idx = nms3dt(ltubes, 0.3)
                alldets[ilabel] += [(v,ltubes[i][1], ltubes[i][0]) for i in idx]
        
        # compute AP for each class
        res = {}
        for ilabel in xrange(d.nlabels):
            detections = alldets[ilabel]
            # load ground-truth
            gt = {}
            for v in vlist:
                tubes = d.gttubes(v)
                
                if not ilabel in tubes:
                    continue
                
                gt[v] = tubes[ilabel]
                
                if len(gt[v])==0:
                    del gt[v]
            
            # precision,recall
            pr = np.empty((len(detections) + 1, 2), dtype=np.float32)
            pr[0,0] = 1.0
            pr[0,1] = 0.0

            fn = sum([ len(g) for g in gt.values()]) # false negatives
            fp = 0 # false positives
            tp = 0 # true positives

            for i, j in enumerate( np.argsort(-np.array([dd[1] for dd in detections]))):
                v, score, tube = detections[j]
                ispositive = False
                
                if v in gt:
                    ious = [iou3dt(g, tube) for g in gt[v]]
                    amax = np.argmax(ious)
                    if ious[amax] >= th:
                        ispositive = True
                        del gt[v][amax]
                        if len(gt[v]) == 0:
                            del gt[v]
                
                if ispositive:
                    tp += 1
                    fn -= 1
                else:
                    fp += 1
                
                pr[i+1,0] = float(tp) / float(tp + fp)
                pr[i+1,1] = float(tp) / float(tp + fn)
            
            res[d.labels[ilabel]] = pr
        
        # save results
        with open(eval_file, 'wb') as fid:
            pickle.dump(res, fid)

    # display results
    ap = 100 * np.array([pr_to_ap(res[label]) for label in d.labels])
    print "frameAP"
    for il, _ in enumerate(d.labels):
        print("{:20s} {:8.2f}".format('', ap[il]))

    print("{:20s} {:8.2f}".format("mAP", np.mean(ap)))
    print("")


if __name__=="__main__":
    exec(sys.argv[1])
