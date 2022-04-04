import mmcv 
import os
datas=mmcv.load('/data/wangqx/DOTA1_0/split_ms_dota1_0/trainval/annfiles/patch_annfile.pkl')

CLASSES=datas['cls']
save_dir= '/data/wangqx/DOTA1_0/split_ms_dota1_0/trainval/annfile/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
for data in datas['content']:
    file_name_txt=data['filename'].replace('.png','.txt')
    with open(os.path.join(save_dir,file_name_txt),'w') as f:
        for i in range(len(data['ann']['labels'])):
            x1=int(data['ann']['bboxes'][i][0])
            y1=int(data['ann']['bboxes'][i][1])
            x2=int(data['ann']['bboxes'][i][2])
            y2=int(data['ann']['bboxes'][i][3])
            x3=int(data['ann']['bboxes'][i][4])
            y3=int(data['ann']['bboxes'][i][5])
            x4=int(data['ann']['bboxes'][i][6])
            y4=int(data['ann']['bboxes'][i][7])
            label=CLASSES[data['ann']['labels'][i]]
            diff=int(data['ann']['diffs'][i])
            import pdb
            pdb.set_trace()
            f.write(x1)
