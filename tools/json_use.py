import json
import os
def resolveJson(json_path):
    file = open(json_path, "rb")
    fileJson = json.load(file)
    path = fileJson["path"]
    outputs = fileJson["outputs"]
    object = outputs["object"]
    name = [object[i]["name"] for i in range(len(object))]
    bndbox = [object[i]["bndbox"] for i in range(len(object))]
    xmin = [bndbox[i]["xmin"] for i in range(len(bndbox))]
    ymin = [bndbox[i]["ymin"] for i in range(len(bndbox))]
    xmax = [bndbox[i]["xmax"] for i in range(len(bndbox))]
    ymax = [bndbox[i]["ymax"] for i in range(len(bndbox))]

    return (path, name, xmin, ymin, xmax,ymax)
def output(path):
    pathname, name, xmin, ymin, xmax,ymax = resolveJson(path)
    f = open("person_label.txt", 'a+', encoding='UTF-8')
    f.write('{} '.format(pathname))
    for i in range(len(name)):
        imgname = int(name[i])
        center_x = int((xmax[i]+xmin[i])/2)
        center_y = int((ymax[i]+ymin[i])/2)
        w = xmax[i]-xmin[i]
        h = ymax[i]-ymin[i]
        f.write('{0} {1} {2} {3} {4} '.format(imgname,center_x,center_y,w,h))
    f.write('\n')
    f.flush()
    f.close()
    return
if __name__ == '__main__':
    path = r"tools/json_files"
    jsondir = os.listdir(path)
    for e in jsondir:
        output(os.path.join(path,e))