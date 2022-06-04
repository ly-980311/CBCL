_base_ = ['./FAIR1M_orcnn_r50_tl_cw_bl8.py']

model = dict(roi_head=dict(bbox_head=dict(class_batch=False, num_of_instance=64)))
