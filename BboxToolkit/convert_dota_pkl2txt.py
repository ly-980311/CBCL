import mmcv 
import os
datas=mmcv.load('/data/wangqx/DOTA1_0/split_ms_dota1_0/trainval/annfiles/patch_annfile.pkl')

CLASSES=datas['cls']
save_dir= '/data/wangqx/DOTA1_0/split_ms_dota1_0/trainval/annfile/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
for data in datas['content']:
    file_name_txt=data['filename'].replace('.png','.txt')
    if file_name_txt=='P1214_0000.txt':

        with open(os.path.join(save_dir,file_name_txt),'w') as f:
            for i in range(len(data['ann']['labels'])):
                x1=str(int(data['ann']['bboxes'][i][0]))+' '
                y1=str(int(data['ann']['bboxes'][i][1]))+' '
                x2=str(int(data['ann']['bboxes'][i][2]))+' '
                y2=str(int(data['ann']['bboxes'][i][3]))+' '
                x3=str(int(data['ann']['bboxes'][i][4]))+' '
                y3=str(int(data['ann']['bboxes'][i][5]))+' '
                x4=str(int(data['ann']['bboxes'][i][6]))+' '
                y4=str(int(data['ann']['bboxes'][i][7]))+' '
                label=CLASSES[data['ann']['labels'][i]]+' '
                diff=str(int(data['ann']['diffs'][i]))

                write_str=x1+y1+x2+y2+x3+y3+x4+y4+label+diff
                f.write(write_str+'\n')
