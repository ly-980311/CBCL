from xml.dom.minidom import Document
import argparse
import json
from PIL import Image
import os


def create_xml(in_dicts, out_path, img_path):
    doc = Document()
    root = doc.createElement('annotation')
    doc.appendChild(root)

    source_list = {'filename': in_dicts['image_name'], 'origin': 'GF2/GF3'}
    node_source = doc.createElement('source')
    for source in source_list:
        node_name = doc.createElement(source)
        node_name.appendChild(doc.createTextNode(source_list[source]))
        node_source.appendChild(node_name)
    root.appendChild(node_source)

    research_list = {'version': '1.0', 'provider': 'FAIR1M', 'author': 'Cyber',
                     'pluginname': 'FAIR1M', 'pluginclass': 'object detection', 'time': '2021-07-21'}
    node_research = doc.createElement('research')
    for research in research_list:
        node_name = doc.createElement(research)
        node_name.appendChild(doc.createTextNode(research_list[research]))
        node_research.appendChild(node_name)
    root.appendChild(node_research)

    img = Image.open(img_path + in_dicts['image_name'])
    size_list = {'width': str(img.size[0]), 'height': str(img.size[1]), 'depth': '3'}
    node_size = doc.createElement('size')
    for size in size_list:
        node_name = doc.createElement(size)
        node_name.appendChild(doc.createTextNode(size_list[size]))
        node_size.appendChild(node_name)
    root.appendChild(node_size)

    node_objects = doc.createElement('objects')
    for i in range(len(in_dicts['labels'])):
        node_object = doc.createElement('object')
        object_fore_list = {'coordinate': 'pixel', 'type': 'rectangle', 'description': 'None'}
        for object_fore in object_fore_list:
            node_name = doc.createElement(object_fore)
            node_name.appendChild(doc.createTextNode(object_fore_list[object_fore]))
            node_object.appendChild(node_name)

        node_possible_result = doc.createElement('possibleresult')
        node_name = doc.createElement('name')
        node_name.appendChild(doc.createTextNode(in_dicts['labels'][i]['category_id']))
        node_possible_result.appendChild(node_name)
        node_object.appendChild(node_possible_result)

        node_points = doc.createElement('points')
        for j in range(4):
            node_point = doc.createElement('point')
            text = '{},{}'.format(in_dicts['labels'][i]['points'][j][0], in_dicts['labels'][i]['points'][j][1])
            node_point.appendChild(doc.createTextNode(text))
            node_points.appendChild(node_point)
        node_point = doc.createElement('point')
        text = '{},{}'.format(in_dicts['labels'][i]['points'][0][0], in_dicts['labels'][i]['points'][0][1])
        node_point.appendChild(doc.createTextNode(text))
        node_points.appendChild(node_point)
        node_object.appendChild(node_points)

        node_objects.appendChild(node_object)
    root.appendChild(node_objects)

    # 开始写xml文档
    filename = out_path + in_dicts['image_name'].split('.')[0] + '.xml'
    fp = open(filename, 'w', encoding='utf-8')
    doc.writexml(fp, indent='', addindent='\t', newl='\n', encoding="utf-8")
    fp.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", help="input file", type=str,
                        default='./output_path/ISPRS_obb_r101_ss_dcn/results_final.json')
    parser.add_argument("--xml_path", help="output path", type=str, default='test/')
    parser.add_argument("--img_path", help="image path", type=str, default='../FAIR1M_origin/test/images/')
    cfg = parser.parse_args()

    json_file = open(cfg.json_path, 'r', encoding='utf-8')
    json_dicts = json.load(json_file)
    if not os.path.exists(cfg.xml_path):
        os.makedirs(cfg.xml_path)
    for json_dict in json_dicts:
        print('Creating {}.xml'.format(json_dict['image_name'].split('.')[0]))
        create_xml(json_dict, cfg.xml_path, cfg.img_path)
    os.system('zip -r test20220322.zip test/')
