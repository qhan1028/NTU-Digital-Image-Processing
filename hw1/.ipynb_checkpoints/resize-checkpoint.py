#
#   DIP Homework 1 - scale.py
#   Written by Liang-Han, 2019.10.18
#

import argparse
import cv2
import os.path as osp


methods = {
    'linear': cv2.INTER_LINEAR,
    'cubic': cv2.INTER_CUBIC
}

parser = argparse.ArgumentParser("Image Resizer")
parser.add_argument("path", type=str, default="imgs/selfie.jpg", help="input image path")
parser.add_argument("--scale", type=float, default=2.0, help="scaling ratio")
parser.add_argument("--method", type=str, choices=list(methods.keys()), default='linear', help='scaling method')
parser.add_argument("--demo", default=False, action="store_true", help="demo homework 1")


def read_image(path):
    print("read image:", path)
    img_bgr = cv2.imread(path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb
    

def write_image(path, img):
    print("write image:", path)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_bgr)
    

def resize_image(img, scale, method='linear'):
    oh, ow, c = img.shape  # original size
    nw, nh = int(ow * scale), int(oh * scale)  # new size
    
    if method in methods:
        return cv2.resize(img, (nw, nh), interpolation=methods[method])
    
    else:
        print('unknown method:', method)
        return img


def demo(path):
    name, ext = osp.splitext(path)
    scales = [0.2, 3.0, 10.0]
    
    img = read_image(path)
    
    for method in methods.keys():
        for scale in scales:
            scaled_img = resize_image(img, scale, method)
            save_path = '-'.join([name, method, '%.1f' % scale]) + ext
            write_image(save_path, scaled_img)


def main(args):
    if args.demo:
        demo(args.path)
        
    else:
        img = read_image(args.path)
        scaled_img = resize_image(img, args.scale, args.method)
        name, ext = osp.splitext(args.path)
        save_path = '-'.join([name, args.method, '%.1f' % args.scale]) + ext
        write_image(save_path, scaled_img)
    
    
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
