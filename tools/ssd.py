
from __future__ import absolute_import
from mxnet import gluon,nd
import gluoncv as gcv
from gluoncv.nn.matcher import CompositeMatcher, BipartiteMatcher, MaximumMatcher

class Prediction(gluon.Block):
    def __init__(self,classes,nms_thresh=0.45, nms_topk=400, post_nms=100):
        super(Prediction,self).__init__()
        self.classes = classes
        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk
        self.post_nms = post_nms
        self.bbox_decoder = gcv.nn.coder.NormalizedBoxCenterDecoder(stds=[0.1,0.1,0.2,0.2])
        self.cls_decoder = gcv.nn.coder.MultiPerClassDecoder(len(self.classes) + 1, thresh=0.01)
        return
    @property
    def num_classes(self):
        return len(self.classes)

    def forward(self, anchors, pred_classes, pred_bboxes):
        """
        :param anchors: (1, num-of-anchor, 4), anchors[0,0,:] = cx,cy,w,h
        :param pred_classes: (batch-size, num-of-anchor, num-of-classes), including background
        :param pred_bboxes: (batch-size, num-of-anchor * 4)

        :param ids: (batch-size, num-of-found, 1)  class id for each found
        :param scores: (batch-size, num-of-found, 1)  class score for each found
        :param bboxes: (batch-size, num-of-found, 4)  coordinates of each found  (x0,y0,x1,y1) with norm w/h
        """
        pred_bboxes = nd.reshape(pred_bboxes, (0,-1,4))
        bboxes = self.bbox_decoder(pred_bboxes, anchors)
        cls_ids, scores = self.cls_decoder(nd.softmax(pred_classes, axis=-1))
        results = []
        for i in range(self.num_classes):
            cls_id = cls_ids.slice_axis(axis=-1, begin=i, end=i + 1)
            score = scores.slice_axis(axis=-1, begin=i, end=i + 1)
            # per class results
            per_result = nd.concat(*[cls_id, score, bboxes], dim=-1)
            results.append(per_result)
        result = nd.concat(*results, dim=1)
        if self.nms_thresh > 0 and self.nms_thresh < 1:
            result = nd.contrib.box_nms(
                result, overlap_thresh=self.nms_thresh, topk=self.nms_topk, valid_thresh=0.01,
                id_index=0, score_index=1, coord_start=2, force_suppress=False)
            if self.post_nms > 0:
                result = result.slice_axis(axis=1, begin=0, end=self.post_nms)
        ids = nd.slice_axis(result, axis=2, begin=0, end=1)
        scores = nd.slice_axis(result, axis=2, begin=1, end=2)
        bboxes = nd.slice_axis(result, axis=2, begin=2, end=6)
        return ids, scores, bboxes


class AssginTarget(gluon.Block):
    def __init__(self):
        super(AssginTarget, self).__init__()
        self.matcher =  CompositeMatcher([BipartiteMatcher(share_max=False), MaximumMatcher(threshold = 0.5)])

        self.center_to_corner = gcv.nn.bbox.BBoxCenterToCorner(split=False)

        #self.sampler = gcv.nn.sampler.NaiveSampler()
        self.sampler = gcv.nn.sampler.OHEMSampler(3.0,thresh=0.5)

        self.cls_encoder = gcv.nn.coder.MultiClassEncoder()
        self.bbox_encoder = gcv.nn.coder.NormalizedBoxCenterEncoder(stds=[0.1,0.1,0.2,0.2])

        return
    def forward(self, anchors, pred_classes, pred_bboxes, groundtruth):
        """
        :param anchors: (1, num-of-anchor, 4), anchors[0,0,:] = cx,cy,w,h
        :param pred_classes: (batch-size, num-of-anchor, num-of-classes), including background
        :param pred_bboxes: (batch-size, num-of-anchor * 4)
        :param groundtruth: (batch-size, max-object-of-one-image, 5), groundtruth[0,0,:] = (cls,x0,y0,x1,y1),
                            (x0,y0,x1,y1) normalized by image size
        :return:
        cls_targets: (batch-size, num-of-anchor, num-of-classes), cls_targets[i,j] = (cls_id+1 for anchor j in image i), including background as class 0
        bbox_targets: (batch-size, num-of-anchor, 4), bbox_targets[i,j,:] = (offset of anchor j in image i) (center mode)
        bbox_masks: (batch-size, num-of-anchor, 4),bbox_mask[i,j,:] = (mask value of anchor j in image i)
        """
        anchors = self.center_to_corner(anchors.reshape(-1,4))
        gt_bboxes = nd.slice_axis(groundtruth,axis=-1, begin=1, end=None)
        gt_classes = nd.slice_axis(groundtruth, axis=-1, begin=0, end=1)
        ious = nd.transpose(nd.contrib.box_iou(anchors, gt_bboxes,format='corner'), (1,0,2))
        matches = self.matcher(ious) #matches: (batch-size, num-of-anchor), matches[i,j] = (idx-of-object in image i matched with anchor j)
        samples = self.sampler(matches, pred_classes, ious) #(batch-size, num-of-anchor), samples[i,j] = -1 or 1

        cls_targets = self.cls_encoder(samples, matches, gt_classes) #(batch-size, num-of-anchor) cls_targets[i,j] = (cls_id+1 for anchor j in image i)
        bbox_targets, bbox_masks = self.bbox_encoder(samples, matches, anchors, gt_bboxes) #(batch-size, num-of-anchor, 4)
                   #bbox_targets[i,j,:] = (offset of anchor j in image i)
                   #bbox_mask[i,j,:] = (mask value of anchor j in image i)
        return cls_targets, bbox_targets, bbox_masks




