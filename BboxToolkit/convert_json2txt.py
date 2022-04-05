import mmcv 
import os
import json

json_dir='../FAIR1M/annotations/'
save_dir= '../FAIR1M/annotations_txt'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for json_name in os.listdir(json_dir):

    json_data=mmcv.load(os.path.join(json_dir,json_name))['shapes']
    txt_name=json_name.replace('json','txt')
    with open(os.path.join(save_dir,txt_name),'w') as f:
        for i in range(len(json_data)):
            x1=str(int(json_data[i]['points'][0][0]))+' '
            y1=str(int(json_data[i]['points'][0][1]))+' '
            x2=str(int(json_data[i]['points'][1][0]))+' '
            y2=str(int(json_data[i]['points'][1][1]))+' '
            x3=str(int(json_data[i]['points'][2][0]))+' '
            y3=str(int(json_data[i]['points'][2][1]))+' '
            x4=str(int(json_data[i]['points'][3][0]))+' '
            y4=str(int(json_data[i]['points'][3][1]))+' '
            label=json_data[i]['label'].replace(' ','-')+' '
            diff='0'
            write_str=x1+y1+x2+y2+x3+y3+x4+y4+label+diff
            f.write(write_str+'\n')
