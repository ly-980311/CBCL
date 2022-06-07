_base_ = ['./FAIR1M_orcnn_r50_tl_cw_bl8.py']

model = dict(roi_head=dict(bbox_head=dict(class_batch=False, add_triplet_loss=False)))
