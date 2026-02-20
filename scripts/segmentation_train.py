
import sys
import argparse
import clip
from torchvision.transforms.transforms import torch
sys.path.append("../")
sys.path.append("./")
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.bratsloader import BRATSDataset, BRATSDataset3D
from guided_diffusion.isicloader import ISICDataset
from guided_diffusion.custom_dataset_loader import CustomDataset
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
import torch as th
from guided_diffusion.train_util import TrainLoop
from visdom import Visdom
viz = Visdom(use_incoming_socket=False)
import torchvision.transforms as transforms
from model import UNET
from transformers import CLIPTextModel

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist(args)
    logger.configure(dir = args.out_dir)

    logger.log("creating data loader...")

    if args.data_name == 'ISIC':
        tran_list = [transforms.Resize((args.image_size,args.image_size)), transforms.ToTensor(),]
        transform_train = transforms.Compose(tran_list)

        ds = ISICDataset(args, args.data_dir, transform_train)
        args.in_ch = 4
    elif args.data_name == 'BRATS':
        tran_list = [transforms.Resize((args.image_size,args.image_size)),]
        transform_train = transforms.Compose(tran_list)

        ds = BRATSDataset3D(args.data_dir, transform_train, test_flag=False)
        args.in_ch = 5
    else :
        tran_list = [transforms.Resize((args.image_size,args.image_size)), transforms.ToTensor(),]
        transform_train = transforms.Compose(tran_list)
        print("Your current directory : ",args.data_dir)
        ds = CustomDataset(args, args.data_dir, transform_train)
        args.in_ch = 4
        
    datal= th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True)
    data = iter(datal)

    logger.log("creating model and diffusion...")

    unet_seg_model = UNET()
    unet_path = './my_checkpoint.pth.tar'
    unet_path1 = torch.load(unet_path, map_location=torch.device('cuda'))

    # print("========================================-------------------------------")
    unet_seg_model.load_state_dict(unet_path1['state_dict'])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # text_model, preprocess = clip.load("ViT-B/32", device=device)

    # Load the text model
    text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    

    # Set the model to evaluation mode
    unet_seg_model.eval()
    # print("============================-------------------------------------------------")
    #unet_seg_model = unet_model.load_state_dict(unet_path1['state_dict'])

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()),unet_seg_model=unet_seg_model,text_model=text_model,
    )
    # print("***************************************************")
    # print(dist_util.dev())
    # print("=================================================")
    if args.multi_gpu:
        model = th.nn.DataParallel(model,find_unused_parameters=True, device_ids=[int(id) for id in args.multi_gpu.split(',')])
        model.to(device = th.device('cuda', int(args.gpu_dev)))
    else:
        # print("==============================================")
        # print(dist_util.dev())
        # print("=================================================")
        model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion,  maxt=args.diffusion_steps)


    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        classifier=None,
        data=data,
        dataloader=datal,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_name = 'ISIC',
        data_dir="../data/ISIC/ISBI2016_ISIC_Part3B_Training_Data",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=1000,
        save_interval=20000,
        resume_checkpoint=None, #"./results/savedmodel040000.pt", #"/results/pretrainedmodel.pt"
        use_fp16=False,
        fp16_scale_growth=1e-3,
        gpu_dev = "0",
        multi_gpu = None, #"0,1,2"
        out_dir='./results/'
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
 