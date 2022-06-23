
_base_ = ['./FAIR1M_orcnn_r50_tl_cw_bl8.py']

model = dict(roi_head=dict(bbox_head=dict(num_of_instance=16)))

# runner = dict(max_epochs=37)
# resume_from = 'work_dirs/max_less_05/latest.pth'
work_dir = 'work_dirs/b16_max_less_1'
