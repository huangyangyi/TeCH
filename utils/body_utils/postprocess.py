import numpy as np
import trimesh
import torch
import argparse
import os.path as osp
import lib.smplx as smplx
import cv2
from pytorch3d.ops import SubdivideMeshes
from pytorch3d.structures import Meshes

from lib.smplx.lbs import general_lbs
from lib.dataset.mesh_util import *
from scipy.spatial import cKDTree
from lib.common.local_affine import register
import json
from lib.common.render_pyrender import PyRender

# loading cfg file
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", type=str, default="exp/demo/teaser/obj/")
parser.add_argument("-n", "--name", type=str, default="")
parser.add_argument("-g", "--gpu", type=int, default=0)
parser.add_argument("-t", "--type", type=str, default="smplx")
parser.add_argument("-f", "--face", action='store_true', default=False)
args = parser.parse_args()

smplx_container = SMPLX()
device = torch.device(f"cuda:{args.gpu}")

prefix = f"{args.dir}/{args.name}"
smpl_path = f"{prefix}_smpl.npy"
tech_path = f"{prefix}_geometry.obj"

smplx_param = np.load(smpl_path, allow_pickle=True).item()
tech_obj = trimesh.load(tech_path)
tech_obj.vertices *= np.array([1.0, -1.0, -1.0])
tech_obj.vertices /= smplx_param["scale"].cpu().numpy()
tech_obj.vertices -= smplx_param["transl"].cpu().numpy()

for key in smplx_param.keys():
    smplx_param[key] = smplx_param[key].to(device).view(1, -1)

smpl_model = smplx.create(
    smplx_container.model_dir,
    model_type=args.type,
    gender="neutral",
    age="adult",
    use_face_contour=False,
    use_pca=False,
    num_betas=200,
    num_expression_coeffs=50,
    flat_hand_mean=True,
    ext='pkl'
).to(device)

smpl_out_lst = []

for pose_type in ["t-pose", "da-pose", "pose", "a-pose"]:
    smpl_out_lst.append(
        smpl_model(
            body_pose=smplx_param["body_pose"],
            global_orient=smplx_param["global_orient"],
            betas=smplx_param["betas"],
            expression=smplx_param["expression"],
            jaw_pose=smplx_param["jaw_pose"],
            left_hand_pose=smplx_param["left_hand_pose"],
            right_hand_pose=smplx_param["right_hand_pose"],
            return_verts=True,
            return_full_pose=True,
            return_joint_transformation=True,
            return_vertex_transformation=True,
            pose_type=pose_type
        )
    )

smpl_verts = smpl_out_lst[2].vertices.detach()[0]
smpl_tree = cKDTree(smpl_verts.cpu().numpy())
dist, idx = smpl_tree.query(tech_obj.vertices, k=5)

if True: #not osp.exists(f"{prefix}_tech_da.obj") or not osp.exists(f"{prefix}_smpl_da.obj"):

    # t-pose for TeCH
    tech_verts = torch.tensor(tech_obj.vertices).float()
    bc_weights, nearest_face = query_barycentric_weights(smpl_verts.new_tensor(tech_obj.vertices[None]), smpl_verts[None], smpl_verts.new_tensor(np.array(smpl_model.faces, dtype=np.int32)[None]))

    ## calculate occlusion map!
    tech_obj_cp = tech_obj.copy()
    tech_obj_cp.vertices += smplx_param["transl"].cpu().numpy()
    tech_obj_cp.vertices *= smplx_param["scale"].cpu().numpy()
    tech_obj_cp.vertices *= np.array([1.0, -1.0, -1.0])
    with open('data/body_data/smplx_vert_segmentation.json') as f:
        smplx_vert_seg = json.load(f)
    seg_labels = list(smplx_vert_seg.keys())
    vert_seg_tensor = torch.zeros(len(smpl_verts), len(seg_labels)).float()
    for k in seg_labels:
        vert_seg_tensor[smplx_vert_seg[k], seg_labels.index(k)] = 1
    tech_vert_seg_weighted = sum([vert_seg_tensor[smpl_model.faces[nearest_face[0]][:, i].astype(np.int32)] * bc_weights[0, :, i].reshape(-1, 1) for i in range(3)])
    seg_max_weight, tech_vert_seg = tech_vert_seg_weighted.max(dim=-1)
    tech_vert_seg[seg_max_weight < 0.5] = -1
    # ['rightHand', 'rightUpLeg', 'leftArm', 'head', 'leftEye', 'rightEye', 'leftLeg', 'leftToeBase', 'leftFoot', 'spine1', 'spine2', 'leftShoulder', 'rightShoulder', 'rightFoot', 'rightArm', 'leftHandIndex1', 'rightLeg', 'rightHandIndex1', 'leftForeArm', 'rightForeArm', 'neck', 'rightToeBase', 'spine', 'leftUpLeg', 'eyeballs', 'leftHand', 'hips']
    L_labels_remove = ['leftArm', 'leftForeArm', 'leftHandIndex1', 'leftShoulder']
    L_labels_occ = ['leftHand', 'leftArm', 'leftForeArm', 'leftHandIndex1']
    R_labels_remove = ['rightArm', 'rightForeArm', 'rightHandIndex1', 'rightShoulder']
    R_labels_occ = ['rightHand', 'rightArm', 'rightForeArm', 'rightHandIndex1']
    def get_occ_map(mesh, vert_label, labels_remove, labels_occ, resolution=1024):
        vert_mask_remove = torch.zeros(len(vert_label)).bool()
        vert_mask_occ = torch.zeros(len(vert_label)).bool()
        for key in labels_remove:
            vert_mask_remove |= (vert_label == seg_labels.index(key))
        for key in labels_occ:
            vert_mask_occ |= (vert_label == seg_labels.index(key))
        mesh_remove = mesh.copy()
        mesh_remove.update_vertices(~vert_mask_remove.cpu().numpy())
        mesh_occ = mesh.copy()
        mesh_occ.update_vertices(vert_mask_occ.cpu().numpy())
        renderer = PyRender(resolution)
        renderer.load_meshes(mesh_remove)
        _, mask_remove = renderer.get_image('front')
        renderer.load_meshes(mesh_occ)
        _, mask_occ = renderer.get_image('front')

        mask_occ = (mask_occ[0] > 0) & (mask_remove[0] > 0)
        return mask_occ.reshape(resolution, resolution)

    left_occ_map = get_occ_map(tech_obj_cp, tech_vert_seg, L_labels_remove, L_labels_occ, 2048)
    #Image.fromarray((left_occ_map * 255).astype(np.uint8)).save('left_occ.png')
    right_occ_map = get_occ_map(tech_obj_cp, tech_vert_seg, R_labels_remove, R_labels_occ, 2048)
    #Image.fromarray((right_occ_map * 255).astype(np.uint8)).save('right_occ.png')
    renderer = PyRender(2048)
    renderer.load_meshes(tech_obj_cp)
    _, loss_mask = renderer.get_image('front')
    loss_mask = loss_mask[0].reshape(2048, 2048)
    occ_mask = (~left_occ_map) & (~right_occ_map)
    kernel = np.ones((20, 20), np.float32)
    erosion_mask = cv2.erode((occ_mask*255).astype(np.uint8), kernel, cv2.BORDER_REFLECT) == 255
    loss_mask = loss_mask & erosion_mask
    Image.fromarray((loss_mask*255).astype(np.uint8)).save(f'{prefix}_occ_mask.png')
    
    bc_weights = torch.tensor(bc_weights).to(device)


    rot_mat_t = sum([smpl_out_lst[2].vertex_transformation.detach()[0][smpl_model.faces_tensor[torch.tensor(nearest_face[0]).long()][:, i]] * bc_weights[0, :, i].reshape(-1, 1, 1) for i in range(3)])
    #rot_mat_t = smpl_out_lst[2].vertex_transformation.detach()[0][idx[:, 0]]
    homo_coord = torch.ones_like(tech_verts)[..., :1]
    tech_cano_verts = torch.inverse(rot_mat_t.cpu()) @ torch.cat([tech_verts, homo_coord],
                                                           dim=1).unsqueeze(-1)
    tech_cano_verts = tech_cano_verts[:, :3, 0].cpu()
    tech_cano = trimesh.Trimesh(tech_cano_verts, tech_obj.faces)

    # da-pose for TeCH
#    rot_mat_da = smpl_out_lst[1].vertex_transformation.detach()[0][idx[:, 0]]
    rot_mat_da = sum([smpl_out_lst[1].vertex_transformation.detach()[0][smpl_model.faces_tensor[torch.tensor(nearest_face[0]).long()][:, i]] * bc_weights[0, :, i].reshape(-1, 1, 1) for i in range(3)])
    tech_da_verts = rot_mat_da.cpu() @ torch.cat([tech_cano_verts, homo_coord], dim=1).unsqueeze(-1)
    tech_da = trimesh.Trimesh(tech_da_verts[:, :3, 0].cpu(), tech_obj.faces)

    # da-pose for SMPL-X
    smpl_da = trimesh.Trimesh(
        smpl_out_lst[1].vertices.detach().cpu()[0], smpl_model.faces, maintain_orders=True, process=False
    )
    smpl_da.export(f"{prefix}_smpl_da.obj")

    # remove hands from TeCH for next registeration
    tech_da_body = tech_da.copy()
    mano_mask = ~np.isin(idx[:, 0], smplx_container.smplx_mano_vid)
    tech_da_body.update_faces(mano_mask[tech_da.faces].all(axis=1))
    tech_da_body.remove_unreferenced_vertices()
    tech_da_body = keep_largest(tech_da_body)

    # remove SMPL-X hand and face
    register_mask = ~np.isin(
        np.arange(smpl_da.vertices.shape[0]),
        np.concatenate([smplx_container.smplx_mano_vid, smplx_container.smplx_front_flame_vid])
    )
    register_mask *= ~smplx_container.eyeball_vertex_mask.bool().numpy()
    smpl_da_body = smpl_da.copy()
    smpl_da_body.update_faces(register_mask[smpl_da.faces].all(axis=1))
    smpl_da_body.remove_unreferenced_vertices()
    smpl_da_body = keep_largest(smpl_da_body)
    
    # upsample the smpl_da_body and do registeration
    smpl_da_body = Meshes(
        verts=[torch.tensor(smpl_da_body.vertices).float()],
        faces=[torch.tensor(smpl_da_body.faces).long()],
    ).to(device)
    sm = SubdivideMeshes(smpl_da_body)
    smpl_da_body = register(tech_da_body, sm(smpl_da_body), device)

    # remove over-streched+hand faces from TeCH
    tech_da_body = tech_da.copy()
    hand_mesh = smpl_da.copy()
    hand_mask = torch.zeros(smplx_container.smplx_verts.shape[0], )
    hand_mask.index_fill_(
        0, torch.tensor(smplx_container.smplx_mano_vid_dict["left_hand"]), 1.0
    )
    hand_mask.index_fill_(
        0, torch.tensor(smplx_container.smplx_mano_vid_dict["right_hand"]), 1.0
    )
    hand_mesh = apply_vertex_mask(hand_mesh, hand_mask)
    tech_da_body = part_removal(tech_da_body, hand_mesh, 8e-2, device=device, smpl_obj=smpl_da, region="hand")
    if args.face:
        face_mesh = smpl_da.copy()
        face_mesh = apply_vertex_mask(face_mesh, smplx_container.front_flame_vertex_mask)
        tech_da_body = part_removal(tech_da_body, face_mesh, 6e-2, device=device, smpl_obj=smpl_da, region="face")
        smpl_face = smpl_da.copy()
        smpl_face.update_faces(smplx_container.front_flame_vertex_mask.numpy()[smpl_face.faces].all(axis=1))
        smpl_face.remove_unreferenced_vertices()
    # stitch the registered SMPL-X body and floating hands to TeCH
    tech_da_tree = cKDTree(tech_da.vertices)
    dist, idx = tech_da_tree.query(smpl_da_body.vertices, k=1)
    smpl_da_body.update_faces((dist > 0.02)[smpl_da_body.faces].all(axis=1))
    smpl_da_body.remove_unreferenced_vertices()

    smpl_hand = smpl_da.copy()
    smpl_hand.update_faces(smplx_container.smplx_mano_vertex_mask.numpy()[smpl_hand.faces].all(axis=1))
    smpl_hand.remove_unreferenced_vertices()
    
    tech_da = [smpl_hand, smpl_da_body, tech_da_body]
    if args.face:
        tech_da.append(smpl_face)
    tech_da = sum(tech_da)
    
    tech_da = poisson(tech_da, f"{prefix}_tech_da.obj", depth=10)

    
else:
    tech_da = trimesh.load(f"{prefix}_tech_da.obj", maintain_orders=True, process=False)
    smpl_da = trimesh.load(f"{prefix}_smpl_da.obj", maintain_orders=True, process=False)


smpl_tree = cKDTree(smpl_da.vertices)
dist, idx = smpl_tree.query(tech_da.vertices, k=5)
knn_weights = np.exp(-dist**2)
knn_weights /= knn_weights.sum(axis=1, keepdims=True)
bc_weights, nearest_face = query_barycentric_weights(torch.tensor(tech_da.vertices).unsqueeze(0).to(device), torch.tensor(smpl_da.vertices).unsqueeze(0).to(device), torch.tensor(np.array(smpl_da.faces, dtype=np.int32)).unsqueeze(0).to(device))
bc_weights = torch.tensor(bc_weights).to(device)

rot_mat_da = sum([smpl_out_lst[1].vertex_transformation.detach()[0][smpl_model.faces_tensor[torch.tensor(nearest_face[0]).long()][:, i]] * bc_weights[0, :, i].reshape(-1, 1, 1) for i in range(3)]).float().cpu()
tech_da_verts = torch.tensor(tech_da.vertices).float()
print('tech_da_verts.shape', tech_da_verts.shape)
tech_cano_verts = torch.inverse(rot_mat_da) @ torch.cat(
    [tech_da_verts, torch.ones_like(tech_da_verts)[..., :1]], dim=1
).unsqueeze(-1)
tech_cano_verts = tech_cano_verts[:, :3, 0].double()
print('tech_cano_verts.shape', tech_cano_verts.shape)

# ----------------------------------------------------
# use any SMPL-X pose to animate TeCH rtechstruction
# ----------------------------------------------------

new_pose = smpl_out_lst[2].full_pose
rot_mat_pose = sum([smpl_out_lst[2].vertex_transformation.detach()[0][smpl_model.faces_tensor[torch.tensor(nearest_face[0]).long()][:, i]] * bc_weights[0, :, i].reshape(-1, 1, 1) for i in range(3)]).float().cpu()
posed_tech_verts = rot_mat_pose @ torch.cat(
    [tech_cano_verts.float(), torch.ones_like(tech_cano_verts.float())[..., :1]], dim=1
).unsqueeze(-1)
posed_tech_verts = posed_tech_verts[:, :3, 0].double()
print('posed_tech_verts.shape', posed_tech_verts.shape)
tech_pose = trimesh.Trimesh(posed_tech_verts.detach(), tech_da.faces)

smplx_param = np.load(smpl_path, allow_pickle=True).item()
tech_pose.vertices += smplx_param["transl"].cpu().numpy()
tech_pose.vertices *= smplx_param["scale"].cpu().numpy()
tech_pose.vertices *= np.array([1.0, -1.0, -1.0])
tech_pose.export(f"{prefix}_pose.obj")

new_pose = smpl_out_lst[3].full_pose
rot_mat_pose = sum([smpl_out_lst[3].vertex_transformation.detach()[0][smpl_model.faces_tensor[torch.tensor(nearest_face[0]).long()][:, i]] * bc_weights[0, :, i].reshape(-1, 1, 1) for i in range(3)]).float().cpu()
posed_tech_verts = rot_mat_pose @ torch.cat(
    [tech_cano_verts.float(), torch.ones_like(tech_cano_verts.float())[..., :1]], dim=1
).unsqueeze(-1)
posed_tech_verts = posed_tech_verts[:, :3, 0].double()
print('aposed_tech_verts.shape', posed_tech_verts.shape)
tech_pose = trimesh.Trimesh(posed_tech_verts.detach(), tech_da.faces)

smplx_param = np.load(smpl_path, allow_pickle=True).item()
tech_pose.vertices += smplx_param["transl"].cpu().numpy()
tech_pose.vertices *= smplx_param["scale"].cpu().numpy()
tech_pose.vertices *= np.array([1.0, 1.0, 1.0])
tech_pose.export(f"{prefix}_apose.obj")
