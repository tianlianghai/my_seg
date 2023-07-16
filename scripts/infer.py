



def infer(image):
    import os
    import sys
    import argparse
    import torch

    # cur_path = os.path.abspath(os.path.dirname(__file__))
    # root_path = os.path.split(cur_path)[0]
    # sys.path.append(root_path)

    from torchvision import transforms
    from PIL import Image
    from core.utils.visualize import get_color_pallete
    from core.models import get_model
    from argparse import Namespace
    # print(args.__dict__)
    args= {'model': 'deeplabv3_resnet50_voc', 'dataset': 'pascal_aug', 'save_folder': '~/.torch/models', 
     'input_pic': '../datasets/voc/VOC2012/JPEGImages/2007_000032.jpg', 'outdir': './eval', 'local_rank': 0, 'aux': False}
    args = Namespace(**args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # image transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    # image = Image.open(config.input_pic).convert('RGB')
    images = transform(image).unsqueeze(0).to(device)

    model = get_model(args.model, pretrained=True, aux=False,  root=args.save_folder).to(device)
    print('Finished loading model!')

    model.eval()
    with torch.no_grad():
        output = model(images)

    pred = torch.argmax(output[0], 1).squeeze(0).cpu().data.numpy()
    mask = get_color_pallete(pred, args.dataset)
    outname = os.path.splitext(os.path.split(args.input_pic)[-1])[0] + '.png'
    mask.save(os.path.join(args.outdir, outname))
    return(mask)


if __name__ == '__main__':
    # demo(args)
    infer("args")
    pass
