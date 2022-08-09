_base_ = ['./FAIR1M_orcnn_r50_tl_cw_bl8.py']

model = dict(roi_head=dict(bbox_head=dict(num_of_instance=32)))
checkpoint_config = dict(interval=1)
# resume_from = 'work_dirs/b32_max_less_07/latest.pth'
work_dir = 'work_dirs/b32_07_t05'
