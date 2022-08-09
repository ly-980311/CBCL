import xml.sax
import os
import json
groups = {'Ship': ['Passenger Ship', 'Motorboat', 'Fishing Boat', 'Tugboat', 'other-ship',
                   'Engineering Ship', 'Liquid Cargo Ship', 'Dry Cargo Ship', 'Warship'],
          'Vehicle': ['Small Car', 'Bus', 'Cargo Truck', 'Dump Truck', 'other-vehicle',
                      'Van', 'Trailer', 'Tractor', 'Excavator', 'Truck Tractor'],
          'Airplane': ['Boeing737', 'Boeing747', 'Boeing777', 'Boeing787', 'ARJ21',
                       'C919', 'A220', 'A321', 'A330', 'A350', 'other-airplane'],
          'Court': ['Baseball Field', 'Basketball Court', 'Football Field', 'Tennis Court'],
          'Road': ['Roundabout', 'Intersection', 'Bridge']}


class GetAnns(xml.sax.ContentHandler):
    def __init__(self):
        self.CurrentData = ''
        self.label = ''
        self.points = []
        self.temp_point = ''
        self.count = 0
        self.shapes = []
        self.shape = {}
        self.content_stack = ''

    def startElement(self, tag, attributes):
        self.CurrentData = tag
        if tag == 'object':
            self.count += 1
            # if self.count == 158:
            #     print('Here!')
            # print(self.count, 'Get an object!')
        self.content_stack = ''

    def endElement(self, tag):
        if tag == 'object':
            self.points = []
            self.shapes.append(self.shape)
            self.shape = {}

        if self.CurrentData == 'name':
            self.label = self.content_stack
            self.shape['label'] = self.label
            for group in groups.keys():
                if self.content_stack in groups[group]:
                    self.shape['group'] = group
        elif self.CurrentData == 'point':
            point = self.content_stack.split(',')
            point = list(map(float, point))
            self.points.append(point)
            if len(self.points) == 6:
                print('{} Error!\n'.format(self.count))
                log.write('{} Error!\n'.format(self.count))
                print(self.points)
                log.write('{}\n'.format(self.points))
            if len(self.points) == 5:
                for p in self.points:
                    if len(p) != 2:
                        print(self.count, self.points.index(p), 'Error!')
                        log.write('{} {} Error!\n'.format(self.count, self.points.index(p)))
                del self.points[-1]
                self.shape['points'] = self.points

        self.CurrentData = ''

    def characters(self, content):
        self.content_stack = self.content_stack + content


if __name__ == '__main__':
    xml_path = '/home/liyan/FAIR1M/part2/labelXml/'
    save_path = '/home/liyan/FAIR1M/part2/ann_txt/'
    log_txt = '/home/liyan/FAIR1M/part2/annotations_log.txt'
    xml_list = os.listdir(xml_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    log = open(log_txt, 'w')
    num = 0
    for xml_name in xml_list:
        if '.xml' in xml_name:
            num += 1
            print(str(num) + " Processing " + xml_name)
            log.write('{} Processing {}\n'.format(num, xml_name))
            xml_file = xml_path + xml_name
            save_file = save_path + xml_name.split('.')[0] + '.txt'
            parser = xml.sax.make_parser()  # create XMLReader
            parser.setFeature(xml.sax.handler.feature_namespaces, 0)  # turn off namespaces
            Handler = GetAnns()  # rewrite ContextHandler
            parser.setContentHandler(Handler)
            parser.parse(xml_file)
            anns = {'shapes': Handler.shapes}
            ann_txt = open(save_file, 'w')
            for ann in anns['shapes']:
                content = ''
                for point in ann['points']:
                    content += '{} {} '.format(str(point[0]), str(point[1]))
                content += '{} 0'.format(ann['label'])
                ann_txt.write('{}\n'.format(content))
            print('Over!')
            log.write('Over!\n')
