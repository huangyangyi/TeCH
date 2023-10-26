# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from lib.dataset.mesh_util import projection
from lib.common.render import Render
from lib.common.render_pyrender import PyRender
import numpy as np
import torch
from torchvision.utils import make_grid
from pytorch3d import _C
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from pytorch3d.structures import Pointclouds
from PIL import Image
import skimage
from thirdparties.lpips import LPIPS
import cv2

from typing import Tuple
from pytorch3d.ops.mesh_face_areas_normals import mesh_face_areas_normals
from pytorch3d.ops.packed_to_padded import packed_to_padded
import copy

_DEFAULT_MIN_TRIANGLE_AREA: float = 5e-3


# PointFaceDistance
class _PointFaceDistance(Function):
    """
    Torch autograd Function wrapper PointFaceDistance Cuda implementation
    """
    @staticmethod
    def forward(
        ctx,
        points,
        points_first_idx,
        tris,
        tris_first_idx,
        max_points,
        min_triangle_area=_DEFAULT_MIN_TRIANGLE_AREA,
    ):
        """
        Args:
            ctx: Context object used to calculate gradients.
            points: FloatTensor of shape `(P, 3)`
            points_first_idx: LongTensor of shape `(N,)` indicating the first point
                index in each example in the batch
            tris: FloatTensor of shape `(T, 3, 3)` of triangular faces. The `t`-th
                triangular face is spanned by `(tris[t, 0], tris[t, 1], tris[t, 2])`
            tris_first_idx: LongTensor of shape `(N,)` indicating the first face
                index in each example in the batch
            max_points: Scalar equal to maximum number of points in the batch
            min_triangle_area: (float, defaulted) Triangles of area less than this
                will be treated as points/lines.
        Returns:
            dists: FloatTensor of shape `(P,)`, where `dists[p]` is the squared
                euclidean distance of `p`-th point to the closest triangular face
                in the corresponding example in the batch
            idxs: LongTensor of shape `(P,)` indicating the closest triangular face
                in the corresponding example in the batch.

            `dists[p]` is
            `d(points[p], tris[idxs[p], 0], tris[idxs[p], 1], tris[idxs[p], 2])`
            where `d(u, v0, v1, v2)` is the distance of point `u` from the triangular
            face `(v0, v1, v2)`

        """
        dists, idxs = _C.point_face_dist_forward(
            points,
            points_first_idx,
            tris,
            tris_first_idx,
            max_points,
            min_triangle_area,
        )
        ctx.save_for_backward(points, tris, idxs)
        ctx.min_triangle_area = min_triangle_area
        return dists, idxs

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_dists):
        grad_dists = grad_dists.contiguous()
        points, tris, idxs = ctx.saved_tensors
        min_triangle_area = ctx.min_triangle_area
        grad_points, grad_tris = _C.point_face_dist_backward(
            points, tris, idxs, grad_dists, min_triangle_area
        )
        return grad_points, None, grad_tris, None, None, None


def _rand_barycentric_coords(
    size1, size2, dtype: torch.dtype, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Helper function to generate random barycentric coordinates which are uniformly
    distributed over a triangle.

    Args:
        size1, size2: The number of coordinates generated will be size1*size2.
                      Output tensors will each be of shape (size1, size2).
        dtype: Datatype to generate.
        device: A torch.device object on which the outputs will be allocated.

    Returns:
        w0, w1, w2: Tensors of shape (size1, size2) giving random barycentric
            coordinates
    """
    uv = torch.rand(2, size1, size2, dtype=dtype, device=device)
    u, v = uv[0], uv[1]
    u_sqrt = u.sqrt()
    w0 = 1.0 - u_sqrt
    w1 = u_sqrt * (1.0 - v)
    w2 = u_sqrt * v
    w = torch.cat([w0[..., None], w1[..., None], w2[..., None]], dim=2)

    return w


def sample_points_from_meshes(meshes, num_samples: int = 10000):
    """
    Convert a batch of meshes to a batch of pointclouds by uniformly sampling
    points on the surface of the mesh with probability proportional to the
    face area.

    Args:
        meshes: A Meshes object with a batch of N meshes.
        num_samples: Integer giving the number of point samples per mesh.
        return_normals: If True, return normals for the sampled points.
        return_textures: If True, return textures for the sampled points.

    Returns:
        3-element tuple containing

        - **samples**: FloatTensor of shape (N, num_samples, 3) giving the
          coordinates of sampled points for each mesh in the batch. For empty
          meshes the corresponding row in the samples array will be filled with 0.
        - **normals**: FloatTensor of shape (N, num_samples, 3) giving a normal vector
          to each sampled point. Only returned if return_normals is True.
          For empty meshes the corresponding row in the normals array will
          be filled with 0.
        - **textures**: FloatTensor of shape (N, num_samples, C) giving a C-dimensional
          texture vector to each sampled point. Only returned if return_textures is True.
          For empty meshes the corresponding row in the textures array will
          be filled with 0.

        Note that in a future releases, we will replace the 3-element tuple output
        with a `Pointclouds` datastructure, as follows

        .. code-block:: python

            Pointclouds(samples, normals=normals, features=textures)
    """
    if meshes.isempty():
        raise ValueError("Meshes are empty.")

    verts = meshes.verts_packed()
    if not torch.isfinite(verts).all():
        raise ValueError("Meshes contain nan or inf.")

    faces = meshes.faces_packed()
    mesh_to_face = meshes.mesh_to_faces_packed_first_idx()
    num_meshes = len(meshes)
    num_valid_meshes = torch.sum(meshes.valid)    # Non empty meshes.

    # Initialize samples tensor with fill value 0 for empty meshes.
    samples = torch.zeros((num_meshes, num_samples, 3), device=meshes.device)

    # Only compute samples for non empty meshes
    with torch.no_grad():
        areas, _ = mesh_face_areas_normals(verts, faces)    # Face areas can be zero.
        max_faces = meshes.num_faces_per_mesh().max().item()
        areas_padded = packed_to_padded(areas, mesh_to_face[meshes.valid], max_faces)    # (N, F)

        # TODO (gkioxari) Confirm multinomial bug is not present with real data.
        samples_face_idxs = areas_padded.multinomial(
            num_samples, replacement=True
        )    # (N, num_samples)
        samples_face_idxs += mesh_to_face[meshes.valid].view(num_valid_meshes, 1)

    # Randomly generate barycentric coords.
    # w                 (N, num_samples, 3)
    # sample_face_idxs  (N, num_samples)
    # samples_verts     (N, num_samples, 3, 3)

    samples_bw = _rand_barycentric_coords(num_valid_meshes, num_samples, verts.dtype, verts.device)
    sample_verts = verts[faces][samples_face_idxs]
    samples[meshes.valid] = (sample_verts * samples_bw[..., None]).sum(dim=-2)

    return samples, samples_face_idxs, samples_bw


def point_mesh_distance(meshes, pcls, weighted=True):

    if len(meshes) != len(pcls):
        raise ValueError("meshes and pointclouds must be equal sized batches")

    # packed representation for pointclouds
    points = pcls.points_packed()    # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]    # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()

    # point to face distance: shape (P,)
    point_to_face, idxs = _PointFaceDistance.apply(
        points, points_first_idx, tris, tris_first_idx, max_points, 5e-3
    )

    if weighted:
        # weight each example by the inverse of number of points in the example
        point_to_cloud_idx = pcls.packed_to_cloud_idx()    # (sum(P_i),)
        num_points_per_cloud = pcls.num_points_per_cloud()    # (N,)
        weights_p = num_points_per_cloud.gather(0, point_to_cloud_idx)
        weights_p = 1.0 / weights_p.float()
        point_to_face = torch.sqrt(point_to_face) * weights_p

    return point_to_face, idxs


class Evaluator:
    def __init__(self, device):

        self.render = Render(size=512, device=device)
        self.render_py = PyRender(size=512, device=device)
        self.device = device
        self.lpips_model = LPIPS(net='vgg').to(device)
    
    @staticmethod
    def psnr_metric(img_pred, img_gt):
        mse = np.mean((img_pred - img_gt)**2)
        psnr = -10 * np.log(mse) / np.log(10)
        return psnr

    @staticmethod
    def scale_for_lpips(image_tensor):
        return image_tensor * 2. - 1.

    def get_lpips_loss(self, rgb, target):
        lpips_loss = self.lpips_model(self.scale_for_lpips(rgb.permute(0, 3, 1, 2).cuda()), 
                        self.scale_for_lpips(target.permute(0, 3, 1, 2).cuda()))
        return torch.mean(lpips_loss).cpu().detach().numpy()
    
    def cal_render_metrics(self, pred, gt, mask=None):
        if mask is not None:
            psnr = self.psnr_metric(pred[mask], gt[mask])
            x, y, w, h = cv2.boundingRect(mask.reshape(gt.shape[:2]).astype(np.uint8)*255)
            pred = pred[y:y + h, x:x + w]
            gt = gt[y:y + h, x:x + w]
            ssim = skimage.metrics.structural_similarity(pred, gt, multichannel=True)
            lpips = self.get_lpips_loss(rgb=torch.from_numpy(pred).float().unsqueeze(0), target=torch.from_numpy(gt).float().unsqueeze(0))
        else:
            psnr = self.psnr_metric(pred, gt)
            ssim = skimage.metrics.structural_similarity(pred, gt, multichannel=True)
            lpips = self.get_lpips_loss(rgb=torch.from_numpy(pred).float().unsqueeze(0), target=torch.from_numpy(gt).float().unsqueeze(0))
        return psnr, ssim, lpips

    def set_mesh(self, result_dict, scale=True):

        # self.verts_uv_gt = None
        # self.verts_uv_pr = None
        # self.uv_image_gt = None
        # self.uv_image_pr = None
        self.verts_color_pr = None
        self.verts_color_gt = None
        for k, v in result_dict.items():
            setattr(self, k, v)
        if scale:
            self.verts_pr -= self.recon_size / 2.0
            self.verts_pr /= self.recon_size / 2.0
        self.verts_gt = projection(self.verts_gt, self.calib)
        self.verts_gt[:, 1] *= -1
        if hasattr(self, 'trimesh_gt') and hasattr(self, 'trimesh_pr'):
            self.trimesh_gt.vertices = self.verts_gt.cpu().numpy()
            self.trimesh_pr.vertices = np.array(self.verts_pr)
        #print('load src')
        self.render.load_meshes(self.verts_pr, self.faces_pr)
        self.src_mesh = self.render.meshes
        # self.src_uv_texture = self.render.uv_textures
        #print('load tgt')
        self.render.load_meshes(self.verts_gt, self.faces_gt)
        self.tgt_mesh = self.render.meshes
        #print(self.src_uv_texture.faces_uvs_padded().shape, self.src_uv_texture.verts_uvs_padded().shape)
        #print(self.tgt_uv_texture.faces_uvs_padded().shape, self.tgt_uv_texture.verts_uvs_padded().shape)

    def calculate_normal_consist(self, normal_path):

        self.render.meshes = self.src_mesh
        src_normal_imgs = self.render.get_image(cam_type="four", bg="black")
        self.render.meshes = self.tgt_mesh
        tgt_normal_imgs = self.render.get_image(cam_type="four", bg="black")

        src_normal_arr = make_grid(torch.cat(src_normal_imgs, dim=0), nrow=4, padding=0)    # [-1,1]
        tgt_normal_arr = make_grid(torch.cat(tgt_normal_imgs, dim=0), nrow=4, padding=0)    # [-1,1]
        src_norm = torch.norm(src_normal_arr, dim=0, keepdim=True)
        tgt_norm = torch.norm(tgt_normal_arr, dim=0, keepdim=True)

        src_mask = (src_norm > 0.9).detach()
        tgt_mask = (tgt_norm > 0.9).detach()
        mask = (tgt_norm > 0.9).detach().cpu().numpy()[0]

        src_norm[src_norm == 0.0] = 1.0
        tgt_norm[tgt_norm == 0.0] = 1.0

        src_normal_arr /= src_norm
        tgt_normal_arr /= tgt_norm

        # sim_mask = self.get_laplacian_2d(tgt_normal_arr).to(self.device)

        src_normal_arr = (src_normal_arr + 1.0) * 0.5
        tgt_normal_arr = (tgt_normal_arr + 1.0) * 0.5

        error = (((src_normal_arr - tgt_normal_arr)**2).sum(dim=0).mean()) * 4.0

        # error_hf = ((((src_normal_arr - tgt_normal_arr) * sim_mask)**2).sum(dim=0).mean()) * 4.0

        normal_img = Image.fromarray(
            (
                torch.cat([torch.cat([src_normal_arr, src_mask.float()], axis=0), torch.cat([tgt_normal_arr, tgt_mask.float()], axis=0)],
                          dim=1).permute(1, 2, 0).detach().cpu().numpy() * 255.0
            ).astype(np.uint8)
        )
        normal_img.save(normal_path)
        psnr, ssim, lpips = 0., 0., 0.
        for i in range(4):
            src_normal_rgb_i = src_normal_arr[..., 512*i:512*(i+1)].permute(1, 2, 0).detach().cpu().numpy()
            tgt_normal_rgb_i = tgt_normal_arr[..., 512*i:512*(i+1)].permute(1, 2, 0).detach().cpu().numpy()
            mask_i = mask[..., 512*i:512*(i+1)]
            psnr_i, ssim_i, lpips_i = self.cal_render_metrics(src_normal_rgb_i, tgt_normal_rgb_i)
            psnr += psnr_i
            ssim += ssim_i
            lpips += lpips_i

        return error, psnr/4, ssim/4, lpips/4

    def calculate_render_consist_cape(self, render_path):
        from lib.dataset.mesh_util import projection_inv, projection
        images_gt = [(img*0.5 + 0.5).clamp(0., 1.).reshape(3, 512, 512).permute(1, 2, 0).cpu().numpy() * 255 for img in self.side_images_gt]
        masks_gt = [mask.clamp(0., 1.).reshape(512, 512).cpu().numpy() for mask in self.side_masks_gt]
        images_gt = [image * mask[..., None] for image, mask in zip(images_gt, masks_gt)]
        images_pr, masks_pr = [], []
        for i in range(3):
            if self.verts_color_pr is None:
                mesh = copy.deepcopy(self.trimesh_pr)
                mesh.vertices = projection(projection_inv(mesh.vertices, self.side_calibs[0]), self.side_calibs[i]).cpu().numpy()
                self.render_py.load_meshes(mesh)
                image_pr, mask_pr = self.render_py.get_image("front")
                image_pr[0] = image_pr[0] * (mask_pr[0][..., None])
                images_pr += image_pr
                masks_pr += mask_pr
            else:
                print('use vertex colors!')
                verts_pr = projection(projection_inv(self.verts_pr, self.side_calibs[0]), self.side_calibs[i])
                self.render.load_meshes(verts_pr.numpy(), self.faces_pr)
                image_pr = self.render.get_image(cam_type="front", bg="black", vertex_colors=self.verts_color_pr)
                image_pr = [image_pr[i].reshape(3, 512, 512).permute(1, 2, 0).cpu().numpy() * 255 for i in range(1)]
                mask_pr = [np.ones_like(image_pr[i][..., 0]) for i in range(1)]
                images_pr += image_pr
                masks_pr += mask_pr
        
        rgb_img = Image.fromarray(
            np.concatenate([
                        np.stack(images_pr, axis=1).reshape([512, 512*3, 3]).astype(np.uint8), 
                        np.stack(images_gt, axis=1).reshape([512, 512*3, 3]).astype(np.uint8), 
            ], axis=0)
        )
        rgb_img.save(render_path)
        psnr, ssim, lpips = [], [], []
        for i in range(3):
            src_rgb_i = images_pr[i] / 255 * masks_pr[i][..., None]
            tgt_rgb_i = images_gt[i] / 255 * masks_gt[i][..., None]
            mask_i = masks_gt[i]
            psnr_i, ssim_i, lpips_i = self.cal_render_metrics(src_rgb_i, tgt_rgb_i)
            vis = src_rgb_i * 0.5 + tgt_rgb_i * 0.5
            vis_img = Image.fromarray(
                (np.concatenate([vis, mask_i.reshape(512, 512, 1)], axis=-1)*255).astype(np.uint8)
            )
            #vis_img.save(render_path.replace('.png', f'_{i}.png'))
            print(psnr_i, ssim_i, lpips_i)
            psnr += [psnr_i]
            ssim += [ssim_i]
            lpips += [lpips_i]

        return psnr, ssim, lpips


    def calculate_render_consist_thuman(self, render_path):
        from lib.dataset.mesh_util import projection_inv, projection
        images_gt = [(img*0.5 + 0.5).clamp(0., 1.).reshape(3, 512, 512).permute(1, 2, 0).cpu().numpy() * 255 for img in self.side_images_gt]
        masks_gt = [mask.clamp(0., 1.).reshape(512, 512).cpu().numpy() for mask in self.side_masks_gt]
        images_gt = [image * mask[..., None] for image, mask in zip(images_gt, masks_gt)]
        images_pr, masks_pr = [], []
        for i in range(4):
            if self.verts_color_pr is None:
                mesh = copy.deepcopy(self.trimesh_pr)
                mesh.vertices = projection(projection_inv(mesh.vertices, self.side_calibs[0]), self.side_calibs[i]).cpu().numpy()
                self.render_py.load_meshes(mesh)
                image_pr, mask_pr = self.render_py.get_image("front")
                image_pr[0] = image_pr[0] * (mask_pr[0][..., None])
                images_pr += image_pr
                masks_pr += mask_pr
            else:
                print('use vertex colors!')
                verts_pr = projection(projection_inv(self.verts_pr, self.side_calibs[0]), self.side_calibs[i])
                self.render.load_meshes(verts_pr.numpy(), self.faces_pr)
                image_pr = self.render.get_image(cam_type="front", bg="black", vertex_colors=self.verts_color_pr)
                image_pr = [image_pr[i].reshape(3, 512, 512).permute(1, 2, 0).cpu().numpy() * 255 for i in range(1)]
                mask_pr = [np.ones_like(image_pr[i][..., 0]) for i in range(1)]
                images_pr += image_pr
                masks_pr += mask_pr
        
        rgb_img = Image.fromarray(
            np.concatenate([
                        np.stack(images_pr, axis=1).reshape([512, 512*4, 3]).astype(np.uint8), 
                        np.stack(images_gt, axis=1).reshape([512, 512*4, 3]).astype(np.uint8), 
            ], axis=0)
        )
        rgb_img.save(render_path)
        psnr, ssim, lpips = [], [], []
        for i in range(4):
            src_rgb_i = images_pr[i] / 255 * masks_pr[i][..., None]
            tgt_rgb_i = images_gt[i] / 255 * masks_gt[i][..., None]
            mask_i = masks_gt[i]
            psnr_i, ssim_i, lpips_i = self.cal_render_metrics(src_rgb_i, tgt_rgb_i)
            vis = src_rgb_i * 0.5 + tgt_rgb_i * 0.5
            vis_img = Image.fromarray(
                (np.concatenate([vis, mask_i.reshape(512, 512, 1)], axis=-1)*255).astype(np.uint8)
            )
            #vis_img.save(render_path.replace('.png', f'_{i}.png'))
            print(psnr_i, ssim_i, lpips_i)
            psnr += [psnr_i]
            ssim += [ssim_i]
            lpips += [lpips_i]

        return psnr, ssim, lpips


    def calculate_render_consist(self, render_path):
        if self.verts_color_gt is None:
            self.render_py.load_meshes(self.trimesh_gt)
            images_gt, masks_gt = self.render_py.get_image("four")
        else:
            print('use vertex colors!')
            self.render.meshes = self.src_mesh
            images_gt = self.render.get_image(cam_type="four", bg="black", vertex_colors=self.verts_color_gt)
            images_gt = [images_gt[i][0].reshape(3, 512, 512).permute(1, 2, 0).cpu().numpy() * 255 for i in range(4)]
            masks_gt = [np.ones_like(images_gt[i][..., 0]) for i in range(4)]
        if self.verts_color_pr is None:
            self.render_py.load_meshes(self.trimesh_pr)
            images_pr, masks_pr = self.render_py.get_image("four")
        else:
            print('use vertex colors!')
            self.render.meshes = self.src_mesh
            images_pr = self.render.get_image(cam_type="four", bg="black", vertex_colors=self.verts_color_pr)
            images_pr = [images_pr[i][0].reshape(3, 512, 512).permute(1, 2, 0).cpu().numpy() * 255 for i in range(4)]
            masks_pr = [np.ones_like(images_pr[i][..., 0]) for i in range(4)]
        
        rgb_img = Image.fromarray(
            np.concatenate([
                        np.stack(images_pr, axis=1).reshape([512, 512*4, 3]).astype(np.uint8), 
                        np.stack(images_gt, axis=1).reshape([512, 512*4, 3]).astype(np.uint8), 
            ], axis=0)
        )
        rgb_img.save(render_path)
        psnr, ssim, lpips = [], [], []
        for i in range(4):
            src_rgb_i = images_pr[i] / 255 * masks_pr[i][..., None]
            tgt_rgb_i = images_gt[i] / 255 * masks_gt[i][..., None]
            mask_i = masks_gt[i]
            psnr_i, ssim_i, lpips_i = self.cal_render_metrics(src_rgb_i, tgt_rgb_i)
            vis = src_rgb_i * 0.5 + tgt_rgb_i * 0.5
            vis_img = Image.fromarray(
                (np.concatenate([vis, mask_i.reshape(512, 512, 1)], axis=-1)*255).astype(np.uint8)
            )
            vis_img.save(render_path.replace('.png', f'_{i}.png'))
            print(psnr_i, ssim_i, lpips_i)
            psnr += [psnr_i]
            ssim += [ssim_i]
            lpips += [lpips_i]

        return psnr, ssim, lpips

    def calculate_chamfer_p2s(self, num_samples=1000):

        samples_tgt, _, _ = sample_points_from_meshes(self.tgt_mesh, num_samples)
        samples_src, _, _ = sample_points_from_meshes(self.src_mesh, num_samples)

        tgt_points = Pointclouds(samples_tgt)
        src_points = Pointclouds(samples_src)

        p2s_dist = point_mesh_distance(self.src_mesh, tgt_points)[0].sum() * 100.0

        chamfer_dist = (
            point_mesh_distance(self.tgt_mesh, src_points)[0].sum() * 100.0 + p2s_dist
        ) * 0.5

        return chamfer_dist, p2s_dist

    def calc_acc(self, output, target, thres=0.5, use_sdf=False):

        # # remove the surface points with thres
        # non_surf_ids = (target != thres)
        # output = output[non_surf_ids]
        # target = target[non_surf_ids]

        with torch.no_grad():
            output = output.masked_fill(output < thres, 0.0)
            output = output.masked_fill(output > thres, 1.0)

            if use_sdf:
                target = target.masked_fill(target < thres, 0.0)
                target = target.masked_fill(target > thres, 1.0)

            acc = output.eq(target).float().mean()

            # iou, precison, recall
            output = output > thres
            target = target > thres

            union = output | target
            inter = output & target

            _max = torch.tensor(1.0).to(output.device)

            union = max(union.sum().float(), _max)
            true_pos = max(inter.sum().float(), _max)
            vol_pred = max(output.sum().float(), _max)
            vol_gt = max(target.sum().float(), _max)

            return acc, true_pos / union, true_pos / vol_pred, true_pos / vol_gt
