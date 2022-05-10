_base_ = ['./orcnn_r50_fpn_1x_ISPRS_3s_le90.py']

fp16 = dict(loss_scale='dynamic')
