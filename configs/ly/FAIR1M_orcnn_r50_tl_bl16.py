_base_ = ['./FAIR1M_orcnn_r50_tl_cw_bl8.py']

model = dict(roi_head=dict(bbox_head=dict(class_batch=False, num_of_instance=16)))

work_dir = 'work_dirs/b16_max_less_07_00_nocw'
