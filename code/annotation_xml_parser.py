import xml.etree.ElementTree as elemTree
import os

CLASSES = ['dog']

def parseFile(path):
    result = ""
    annotation = elemTree.parse(path)

    filename = annotation.find('filename')
    size = annotation.find('size')
    width = size.find('width')
    height = size.find('height')

    result += filename.text + ' ' + width.text + ' ' + height.text
    for obj in annotation.findall('./object'):
        name = obj.find('name')
        box = obj.find('bndbox')

        label_index = CLASSES.index(name.text)
        xmin = box.find('xmin')
        ymin = box.find('ymin')
        xmax = box.find('xmax')
        ymax = box.find('ymax')

        result += ' ' + str(label_index) + ' ' + xmin.text + ' ' + ymin.text + ' ' + xmax.text + ' ' + ymax.text

    return result
        

def parseDir(path):
    result = ""
    for file in os.listdir(path):
        if file.split('.')[-1] != 'xml':
            continue
        
        result += parseFile(path+'/'+file) + '\n'
    return result


directory = input('Input directory, or 0 to exit: ')

if directory != '0':
    result = parseDir(directory)
    file = open(directory+'/'+'annotation.txt','w')
    file.write(result)
    file.close()
    print("Complete!")
