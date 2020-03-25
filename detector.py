from darknet53 import *
import cfg
from PIL import Image,ImageDraw,ImageFont
from torchvision import transforms
import os
import torch
from tools.utils import *

class Detector:

    def __init__(self):
        self.save_path = 'models/700net.pt'
        self.net = MainNet().cuda()
        if os.path.exists(self.save_path):
            self.net.load_state_dict(torch.load(self.save_path))

    def detector(self, input, thresh, anchors):
        self.net.eval()
        _boxes = []
        output_13, output_26, output_52 = self.net(input)

        idxs_13, vecs_13 = self._filter(output_13, thresh)
        boxes_13 = self._parse(idxs_13, vecs_13, 32, anchors[13])

        idxs_26, vecs_26 = self._filter(output_26, thresh)
        boxes_26 = self._parse(idxs_26, vecs_26, 16, anchors[26])

        idxs_52, vecs_52 = self._filter(output_52, thresh)
        boxes_52 = self._parse(idxs_52, vecs_52, 8, anchors[52])

        boxes = torch.cat([boxes_13, boxes_26, boxes_52], dim=0)
        boxes = boxes.cpu().detach().numpy()
        if boxes.shape[0]==0:
            return np.array([])
        for i in range(5):
            box = boxes[boxes[:,5] == i]
            if box.shape[0] == 0:
                continue
            else:
                _boxes.extend(nms(box,0.3))
        return np.stack(_boxes).reshape(-1,6)

    def _filter(self, output, thresh):

        output = output.permute(0, 2, 3, 1)
        output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)
        output1 = torch.sigmoid(output[..., :3])
        mask = output1[..., 0] > thresh
        idxs = mask.nonzero()
        vecs1 = output1[mask]
        vecs = output[mask][:,3:]
        vecs = torch.cat((vecs1,vecs),dim=1)
        return idxs, vecs

    def _parse(self, idxs, vecs, t, anchors):
        if not idxs.size(0) == 0:
            anchors = torch.Tensor(anchors).cuda()
            # n = idxs[:, 0]  # 所属的图片
            cls = vecs[:, 0]  # 置信度
            a = idxs[:, 3]  # 建议框

            cy = (idxs[:, 1].float() + vecs[:, 2]) * t  # 原图的中心点y
            cx = (idxs[:, 2].float() + vecs[:, 1]) * t  # 原图的中心点x

            w = anchors[a, 0] * torch.exp(vecs[:, 3])
            h = anchors[a, 1] * torch.exp(vecs[:, 4])
            category = torch.argmax(vecs[:,5:],dim=1).float()
            return torch.stack([cls, cx, cy, w, h, category], dim=1)
        return torch.tensor([]).cuda()

if __name__ == '__main__':
    # obj = Detector()
    # image_file = "data/img/3.jpg"
    # im =  Image.open(image_file)
    #
    # font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 18)
    # img = transforms.ToTensor()(im).unsqueeze(0).cuda()
    # boxes = obj.detector(img, 0.6, cfg.ANCHORS_GROUP)
    # imDraw = ImageDraw.Draw(im)
    # if boxes.shape[0] == 0:
    #     print("no")
    # else:
    #     for box in boxes:
    #         x1 = int(box[1]-box[3]/2)
    #         y1 = int(box[2]-box[4]/2)
    #         x2 = int(box[1]+box[3]/2)
    #         y2 = int(box[2]+box[4]/2)
    #         print(box[0],int(box[5]))
    #         imDraw.rectangle((x1, y1, x2, y2), outline='red')
    #         imDraw.text((max(x1,0) + 3, max(y1,0) + 3), fill=(0, 255, 0), text=str(int(box[5])), font=font)
    #     # im.save('2.jpg')
    #     im.show()

    obj = Detector()
    path = r'C:\Users\A\Desktop\img'
    imgdir = os.listdir(path)
    for i, e in enumerate(imgdir):
        im = Image.open(os.path.join(path, e))
        img = im.convert("RGB")
        w,h = img.size
        max_side = max(w,h)
        img = img.crop((0,0,max_side,max_side))
        img = img.resize((416,416))
        d = max_side/416
        font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 40)
        img = transforms.ToTensor()(img).unsqueeze(0).cuda()
        boxes = obj.detector(img, 0.8, cfg.ANCHORS_GROUP)
        imDraw = ImageDraw.Draw(im)
        if boxes.shape[0] == 0:
            print("no")
        else:
            for box in boxes:
                x1 = int((box[1] - box[3] / 2)*d)
                y1 = int((box[2] - box[4] / 2)*d)
                x2 = int((box[1] + box[3] / 2)*d)
                y2 = int((box[2] + box[4] / 2)*d)
                print(box[0], int(box[5]))
                imDraw.rectangle((x1, y1, x2, y2), outline='red',width=5)
                imDraw.text((max(x1, 0) + 3, max(y1, 0) + 3), fill=(0, 255, 0), text=str(int(box[5])), font=font)
            im.save(r'C:\Users\A\Desktop\img2\{}.jpg'.format(i))
            # im.show()
