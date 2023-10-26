#import nvdiffrast.torch as dr
import torch
import argparse

from lib.provider import ViewDataset
from lib.trainer import *
from lib.renderer import Renderer

from yacs.config import CfgNode as CN


def load_config(path, default_path=None):
    cfg = CN(new_allowed=True)
    if default_path is not None:
        cfg.merge_from_file(default_path)
    cfg.merge_from_file(path)

    return cfg
#torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="config file")
    parser.add_argument('--exp_dir', type=str, required=True, help="experiment dir")
    parser.add_argument('--sub_name', type=str, required=True, help="subject name")
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    parser.add_argument('--test', action="store_true")


    opt = parser.parse_args()
    cfg = load_config(opt.config, default_path="configs/default.yaml")
    cfg.test.test = opt.test
    cfg.workspace = os.path.join(opt.exp_dir, cfg.stage)
    cfg.exp_root = opt.exp_dir
    cfg.sub_name = opt.sub_name
    if cfg.data.load_input_image:
        cfg.data.img = os.path.join(opt.exp_dir, 'png', "{}_crop.png".format(opt.sub_name))
    if cfg.data.load_front_normal:
        cfg.data.front_normal_img = os.path.join(opt.exp_dir, 'normal', "{}_normal_front.png".format(opt.sub_name))
    if cfg.data.load_back_normal:
        cfg.data.back_normal_img = os.path.join(opt.exp_dir, 'normal', "{}_normal_back.png".format(opt.sub_name))
    if cfg.data.load_keypoints:
        cfg.data.keypoints_path = os.path.join(opt.exp_dir, 'obj', "{}_smpl.npy".format(opt.sub_name))
    if cfg.data.load_result_mesh:
        cfg.data.last_model = os.path.join(opt.exp_dir, 'obj', "{}_pose.obj".format(opt.sub_name))
        cfg.data.last_ref_model = os.path.join(opt.exp_dir, 'obj', "{}_smpl.obj".format(opt.sub_name))
    else:
        cfg.data.last_model = os.path.join(opt.exp_dir, 'obj', "{}_smpl.obj".format(opt.sub_name))
    if cfg.data.load_apose_mesh:
        cfg.data.can_pose_folder = os.path.join(opt.exp_dir, 'obj', "{}_apose.obj".format(opt.sub_name))
    if cfg.data.load_apose_mesh:
        cfg.data.can_pose_folder = os.path.join(opt.exp_dir, 'obj', "{}_apose.obj".format(opt.sub_name))
    if cfg.data.load_occ_mask:
        cfg.data.occ_mask = os.path.join(opt.exp_dir, 'png', "{}_occ_mask.png".format(opt.sub_name))
    if cfg.data.load_da_pose_mesh:
        cfg.data.da_pose_mesh = os.path.join(opt.exp_dir, 'obj', "{}_da_pose.obj".format(opt.sub_name))
    if cfg.guidance.use_dreambooth:
        cfg.guidance.hf_key = os.path.join(opt.exp_dir, 'sd_model')
    if cfg.guidance.text is None:
        with open(os.path.join(opt.exp_dir, 'prompt.txt'), 'r') as f:
            cfg.guidance.text = f.readlines()[0].split('|')[0]

    print(cfg)

    seed_everything(opt.seed)
    model = Renderer(cfg)
    if model.keypoints is not None:
        if len(model.keypoints[0]) == 1:
            cfg.train.head_position = model.keypoints[0][0].cpu().numpy().tolist()
        else:
            cfg.train.head_position = model.keypoints[0][15].cpu().numpy().tolist()
    else:
        cfg.train.head_position = np.array([0., 0.4, 0.], dtype=np.float32).tolist()
    cfg.train.canpose_head_position = np.array([0., 0.4, 0.], dtype=np.float32).tolist()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if cfg.test.test:
        guidance = None  # no need to load guidance model at test
        trainer = Trainer(
            'df', cfg, model, guidance, device=device, workspace=cfg.workspace, fp16=cfg.fp16, use_checkpoint=cfg.train.ckpt, pretrained=cfg.train.pretrained)

        if not cfg.test.not_test_video:
            test_loader = ViewDataset(cfg, device=device, type='test', H=cfg.test.H, W=cfg.test.W, size=100, render_head=True).dataloader()
            trainer.test(test_loader, write_image=cfg.test.write_image)
            if cfg.data.can_pose_folder is not None:
                trainer.test(test_loader, write_image=cfg.test.write_image, can_pose=True)  
        if cfg.test.save_mesh:
            trainer.save_mesh()
    else:

        train_loader = ViewDataset(cfg, device=device, type='train', H=cfg.train.h, W=cfg.train.w, size=100).dataloader()
        params_list = list()
        if cfg.guidance.type == 'stable-diffusion':
            from lib.guidance import StableDiffusion
            guidance = StableDiffusion(device, cfg.guidance.sd_version, cfg.guidance.hf_key, cfg.guidance.step_range, controlnet=cfg.guidance.controlnet, lora=cfg.guidance.lora, cfg=cfg, head_hf_key=cfg.guidance.head_hf_key)
            for p in guidance.parameters():
                p.requires_grad = False
        else:
            raise NotImplementedError(f'--guidance {cfg.guidance.type} is not implemented.')
        
        if cfg.train.optim == 'adan':
            from lib.optimizer import Adan
            # Adan usually requires a larger LR
            params_list.extend(model.get_params(5 * cfg.train.lr))
            optimizer = lambda model: Adan(
                params_list, eps=1e-8, weight_decay=2e-5, max_grad_norm=5.0, foreach=False)
        else:  # adam
            params_list.extend(model.get_params(cfg.train.lr))
            optimizer = lambda model: torch.optim.Adam(params_list, betas=(0.9, 0.99), eps=1e-15)

        # scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 1) # fixed
        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1**min(iter / cfg.train.iters, 1))

        trainer = Trainer(
            'df',
            cfg,
            model,
            guidance,
            device=device,
            workspace=cfg.workspace,
            optimizer=optimizer,
            ema_decay=None,
            fp16=cfg.train.fp16,
            lr_scheduler=scheduler,
            use_checkpoint=cfg.train.ckpt,
            eval_interval=cfg.train.eval_interval,
            scheduler_update_every_step=True, 
            pretrained=cfg.train.pretrained)

        valid_loader = ViewDataset(cfg, device=device, type='val', H=cfg.test.H, W=cfg.test.W, size=5).dataloader()

        max_epoch = np.ceil(cfg.train.iters / len(train_loader)).astype(np.int32)
        if cfg.profile:
            import cProfile
            with cProfile.Profile() as pr:
                trainer.train(train_loader, valid_loader, max_epoch)        
                pr.dump_stats(os.path.join(cfg.workspace, 'profile.dmp'))
                pr.print_stats()
        else:
            trainer.train(train_loader, valid_loader, max_epoch)

        test_loader = ViewDataset(cfg, device=device, type='test', H=cfg.test.H, W=cfg.test.W, size=100, render_head=True).dataloader()
        trainer.test(test_loader, write_image=cfg.test.write_image)

        if cfg.test.save_mesh:
            trainer.save_mesh()