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

import math
import os

import cv2
import numpy as np
import torch
from PIL import ImageColor
from pytorch3d.renderer import look_at_view_transform
#from pytorch3d.renderer.mesh import TexturesVertex, TexturesUV
#from pytorch3d.structures import Meshes
import pyrender
from termcolor import colored
from tqdm import tqdm

import lib.common.render_utils as util
from lib.common.imutils import blend_rgb_norm
from lib.dataset.mesh_util import get_visibility
import trimesh
import pyrr 

def image2vid(images, vid_path):

    os.makedirs(os.path.dirname(vid_path), exist_ok=True)

    w, h = images[0].size
    videodims = (w, h)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video = cv2.VideoWriter(vid_path, fourcc, len(images) / 5.0, videodims)
    for image in images:
        video.write(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
    video.release()


class PyRender:
    def __init__(self, size=512, ssaa=8, device=torch.device("cuda:0")):
        self.device = device
        self.size = size

        # camera setting
        self.dis = 1.0
        self.scale = 1.0
        self.mesh_y_center = 0.0

        # speed control
        self.fps = 30
        self.step = 3

        self.cam_pos = {
            "front":
            torch.tensor([
                (0, self.mesh_y_center, self.dis),
            ]), "frontback":
            torch.tensor([
                (0, self.mesh_y_center, self.dis),
                (0, self.mesh_y_center, -self.dis),
            ]), "four":
            torch.tensor([
                (0, self.mesh_y_center, self.dis),
                (self.dis, self.mesh_y_center, 0),
                (0, self.mesh_y_center, -self.dis),
                (-self.dis, self.mesh_y_center, 0),
            ]), "around":
            torch.tensor([(
            100.0 * math.cos(np.pi / 180 * angle), self.mesh_y_center,
                100.0 * math.sin(np.pi / 180 * angle)
            ) for angle in range(0, 360, self.step)])
        }

        self.type = "color"

        self.meshes = None
        self.deform_mesh = None
        self.pcd = None
        self.renderer = pyrender.OffscreenRenderer(self.size * ssaa, self.size * ssaa)

    def get_camera_poses(self, type="four", idx=None):

        if idx is None:
            idx = np.arange(len(self.cam_pos[type]))

        R, T = look_at_view_transform(
            eye=self.cam_pos[type][idx],
            at=((0, self.mesh_y_center, 0), ),
            up=((0, 1, 0), ),
        )

        camera_poses = []
        for i in idx:
            camera_poses.append(np.linalg.inv(np.array(pyrr.Matrix44.look_at(
                eye=self.cam_pos[type][i],
                target=np.array([0, self.mesh_y_center, 0]),
                up=np.array([0, 1, 0])
            )).T))
            # camera_mats[i][:3, :3] = R[i].T
            # camera_mats[i][:3, 3] = -R[i].T @ T[i]
        return camera_poses


    def load_meshes(self, mesh: trimesh.Trimesh):
        """load mesh into the pytorch3d renderer

        Args:
            verts ([N,3] / [B,N,3]): array or tensor
            faces ([N,3]/ [B,N,3]): array or tensor
        """
        #print('1', faces.shape, verts_uv.shape)
        # primitive_material = pyrender.material.MetallicRoughnessMaterial(
        #         alphaMode='BLEND',
        #         baseColorFactor=[0.3, 0.3, 0.3, 1.0],
        #         metallicFactor=0.8, 
        #         roughnessFactor=0.8 
        #     )
        self.meshes = pyrender.Mesh.from_trimesh(mesh, smooth=False) #, material=primitive_material)

    def get_image(self, cam_type="frontback", type="rgb", bg="gray", uv_textures=False):
        camera_poses = self.get_camera_poses(cam_type)
        images = []
        masks = []
        for pose in camera_poses:
            scene = pyrender.Scene(ambient_light=(1.0,1.0,1.0))
            scene.add(self.meshes)
            camera = pyrender.OrthographicCamera(1., 1.)
            #camera_node = pyrender.Node(camera=camera, matrix=mat)
            scene.add(camera, pose=pose)
            color, depth = self.renderer.render(scene, flags=pyrender.constants.RenderFlags.SKIP_CULL_FACES | pyrender.constants.RenderFlags.ALL_SOLID | pyrender.constants.RenderFlags.SKIP_CULL_FACES)
            print(color.shape, depth.shape)
            mask = depth > 0
            color = cv2.resize(color, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
            mask = cv2.resize((mask*255).astype(np.uint8), (self.size, self.size), interpolation=cv2.INTER_CUBIC) == 255
            images.append(color)
            masks.append(mask)
        return images, masks
    
    def get_image_depth(self, cam_type="frontback", type="rgb", bg="gray", uv_textures=False):
        camera_poses = self.get_camera_poses(cam_type)
        images = []
        masks = []
        depths = []
        for pose in camera_poses:
            scene = pyrender.Scene(ambient_light=(1.0,1.0,1.0))
            scene.add(self.meshes)
            camera = pyrender.OrthographicCamera(1., 1.)
            #camera_node = pyrender.Node(camera=camera, matrix=mat)
            scene.add(camera, pose=pose)
            color, depth = self.renderer.render(scene, flags=pyrender.constants.RenderFlags.SKIP_CULL_FACES | pyrender.constants.RenderFlags.ALL_SOLID | pyrender.constants.RenderFlags.SKIP_CULL_FACES)
            mask = depth > 0
            images.append(color)
            masks.append(mask)
            depths.append(depth)
        return images, masks, depths
 