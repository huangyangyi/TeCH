import torch
import torch.nn as nn
import kaolin as kal
from tqdm import tqdm
import random
import trimesh
from .network_utils import Decoder, HashDecoder, HashDecoderNew
# Laplacian regularization using umbrella operator (Fujiwara / Desbrun).
# https://mgarland.org/class/geom04/material/smoothing.pdf
def laplace_regularizer_const(mesh_verts, mesh_faces):
    term = torch.zeros_like(mesh_verts)
    norm = torch.zeros_like(mesh_verts[..., 0:1])

    v0 = mesh_verts[mesh_faces[:, 0], :]
    v1 = mesh_verts[mesh_faces[:, 1], :]
    v2 = mesh_verts[mesh_faces[:, 2], :]

    term.scatter_add_(0, mesh_faces[:, 0:1].repeat(1,3), (v1 - v0) + (v2 - v0))
    term.scatter_add_(0, mesh_faces[:, 1:2].repeat(1,3), (v0 - v1) + (v2 - v1))
    term.scatter_add_(0, mesh_faces[:, 2:3].repeat(1,3), (v0 - v2) + (v1 - v2))

    two = torch.ones_like(v0) * 2.0
    norm.scatter_add_(0, mesh_faces[:, 0:1], two)
    norm.scatter_add_(0, mesh_faces[:, 1:2], two)
    norm.scatter_add_(0, mesh_faces[:, 2:3], two)

    term = term / torch.clamp(norm, min=1.0)

    return torch.mean(term**2)

def loss_f(mesh_verts, mesh_faces, points, it):
    pred_points = kal.ops.mesh.sample_points(mesh_verts.unsqueeze(0), mesh_faces, 50000)[0][0]
    chamfer = kal.metrics.pointcloud.chamfer_distance(pred_points.unsqueeze(0), points.unsqueeze(0)).mean()
    laplacian_weight = 0.1
    if it > iterations//2:
        lap = laplace_regularizer_const(mesh_verts, mesh_faces)
        return chamfer + lap * laplacian_weight
    return chamfer

###############################################################################
# Compact tet grid
###############################################################################

def compact_tets(pos_nx3, sdf_n, tet_fx4):
    with torch.no_grad():
        # Find surface tets
        occ_n = sdf_n > 0
        occ_fx4 = occ_n[tet_fx4.reshape(-1)].reshape(-1, 4)
        occ_sum = torch.sum(occ_fx4, -1)
        valid_tets = (occ_sum > 0) & (occ_sum < 4)  # one value per tet, these are the surface tets

        valid_vtx = tet_fx4[valid_tets].reshape(-1)
        unique_vtx, idx_map = torch.unique(valid_vtx, dim=0, return_inverse=True)
        new_pos = pos_nx3[unique_vtx]
        new_sdf = sdf_n[unique_vtx]
        new_tets = idx_map.reshape(-1, 4)
        return new_pos, new_sdf, new_tets


###############################################################################
# Subdivide volume
###############################################################################

def sort_edges(edges_ex2):
    with torch.no_grad():
        order = (edges_ex2[:, 0] > edges_ex2[:, 1]).long()
        order = order.unsqueeze(dim=1)
        a = torch.gather(input=edges_ex2, index=order, dim=1)
        b = torch.gather(input=edges_ex2, index=1 - order, dim=1)
    return torch.stack([a, b], -1)


def batch_subdivide_volume(tet_pos_bxnx3, tet_bxfx4):
    device = tet_pos_bxnx3.device
    # get new verts
    tet_fx4 = tet_bxfx4[0]
    edges = [0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3]
    all_edges = tet_fx4[:, edges].reshape(-1, 2)
    all_edges = sort_edges(all_edges)
    unique_edges, idx_map = torch.unique(all_edges, dim=0, return_inverse=True)
    idx_map = idx_map + tet_pos_bxnx3.shape[1]
    all_values = tet_pos_bxnx3
    mid_points_pos = all_values[:, unique_edges.reshape(-1)].reshape(
        all_values.shape[0], -1, 2,
        all_values.shape[-1]).mean(2)
    new_v = torch.cat([all_values, mid_points_pos], 1)

    # get new tets

    idx_a, idx_b, idx_c, idx_d = tet_fx4[:, 0], tet_fx4[:, 1], tet_fx4[:, 2], tet_fx4[:, 3]
    idx_ab = idx_map[0::6]
    idx_ac = idx_map[1::6]
    idx_ad = idx_map[2::6]
    idx_bc = idx_map[3::6]
    idx_bd = idx_map[4::6]
    idx_cd = idx_map[5::6]

    tet_1 = torch.stack([idx_a, idx_ab, idx_ac, idx_ad], dim=1)
    tet_2 = torch.stack([idx_b, idx_bc, idx_ab, idx_bd], dim=1)
    tet_3 = torch.stack([idx_c, idx_ac, idx_bc, idx_cd], dim=1)
    tet_4 = torch.stack([idx_d, idx_ad, idx_cd, idx_bd], dim=1)
    tet_5 = torch.stack([idx_ab, idx_ac, idx_ad, idx_bd], dim=1)
    tet_6 = torch.stack([idx_ab, idx_ac, idx_bd, idx_bc], dim=1)
    tet_7 = torch.stack([idx_cd, idx_ac, idx_bd, idx_ad], dim=1)
    tet_8 = torch.stack([idx_cd, idx_ac, idx_bc, idx_bd], dim=1)

    tet_np = torch.cat([tet_1, tet_2, tet_3, tet_4, tet_5, tet_6, tet_7, tet_8], dim=0)
    tet_np = tet_np.reshape(1, -1, 4).expand(tet_pos_bxnx3.shape[0], -1, -1)
    tet = tet_np.long().to(device)

    return new_v, tet


class DMTetMesh(nn.Module):
    def __init__(self, vertices: torch.Tensor, indices: torch.Tensor, device: str='cuda', grid_scale=1e-4, use_explicit=False, geo_network='mlp', hash_max_res=1024, hash_num_levels=16, num_subdiv=0) -> None:
        super().__init__()
        self.device = device
        self.tet_v = vertices.to(device)
        self.tet_ind = indices.to(device)
        self.use_explicit = use_explicit
        if self.use_explicit:
            self.sdf = nn.Parameter(torch.zeros_like(self.tet_v[:, 0]), requires_grad=True)
            self.deform = nn.Parameter(torch.zeros_like(self.tet_v), requires_grad=True)
        elif geo_network == 'mlp':
            self.decoder = Decoder().to(device)
        elif geo_network == 'hash':
            pts_bounds = (self.tet_v.min(dim=0)[0], self.tet_v.max(dim=0)[0])
            self.decoder = HashDecoder(input_bounds=pts_bounds, max_res=hash_max_res, num_levels=hash_num_levels).to(device)
        self.grid_scale = grid_scale
        self.num_subdiv = num_subdiv

    def query_decoder(self, tet_v):
        if self.tet_v.shape[0] < 1000000:
            return self.decoder(tet_v)
        else:
            chunk_size = 1000000
            results = []
            for i in range((tet_v.shape[0] // chunk_size) + 1):
                if i*chunk_size < tet_v.shape[0]:
                    results.append(self.decoder(tet_v[i*chunk_size: (i+1)*chunk_size]))
            return torch.cat(results, dim=0)

    def get_mesh(self, return_loss=False, num_subdiv=None):
        if num_subdiv is None:
            num_subdiv = self.num_subdiv
        if self.use_explicit:
            sdf = self.sdf * 1
            deform = self.deform * 1
        else:
            pred = self.query_decoder(self.tet_v)
            sdf, deform = pred[:,0], pred[:,1:]
        verts_deformed = self.tet_v + torch.tanh(deform) * self.grid_scale / 2 # constraint deformation to avoid flipping tets
        tet = self.tet_ind
        for i in range(num_subdiv):
            verts_deformed, _, tet = compact_tets(verts_deformed, sdf, tet)
            verts_deformed, tet = batch_subdivide_volume(verts_deformed.unsqueeze(0), tet.unsqueeze(0))
            verts_deformed = verts_deformed[0]
            tet = tet[0]
            pred = self.query_decoder(verts_deformed)
            sdf, _ = pred[:,0], pred[:,1:]
        mesh_verts, mesh_faces = kal.ops.conversions.marching_tetrahedra(verts_deformed.unsqueeze(0), tet, sdf.unsqueeze(0)) # running MT (batched) to extract surface mesh
            
        mesh_verts, mesh_faces = mesh_verts[0], mesh_faces[0]
        return mesh_verts, mesh_faces, None
    
    def init_mesh(self, mesh_v, mesh_f, init_padding=0.):
        num_pts = self.tet_v.shape[0]
        mesh = trimesh.Trimesh(mesh_v.cpu().numpy(), mesh_f.cpu().numpy())
        import mesh_to_sdf
        sdf_tet = torch.tensor(mesh_to_sdf.mesh_to_sdf(mesh, self.tet_v.cpu().numpy()), dtype=torch.float32).to(self.device) - init_padding
        sdf_mesh_v, sdf_mesh_f = kal.ops.conversions.marching_tetrahedra(self.tet_v.unsqueeze(0), self.tet_ind, sdf_tet.unsqueeze(0))
        sdf_mesh_v, sdf_mesh_f = sdf_mesh_v[0], sdf_mesh_f[0]
        if self.use_explicit:
            self.sdf.data[...] = sdf_tet[...]
        else:
            optimizer = torch.optim.Adam(list(self.parameters()), lr=1e-3)
            batch_size = 300000
            iter = 1000
            points, sdf_gt = mesh_to_sdf.sample_sdf_near_surface(mesh) 
            valid_idx = (points < self.tet_v.cpu().numpy().min(axis=0)).sum(-1) + (points > self.tet_v.cpu().numpy().max(axis=0)).sum(-1) == 0
            points = points[valid_idx]
            sdf_gt = sdf_gt[valid_idx]
            points = torch.tensor(points, dtype=torch.float32).to(self.device)
            sdf_gt = torch.tensor(sdf_gt, dtype=torch.float32).to(self.device)
            points = torch.cat([points, self.tet_v], dim=0)
            sdf_gt = torch.cat([sdf_gt, sdf_tet], dim=0)
            num_pts = len(points)
            for i in tqdm(range(iter)):
                sampled_ind = random.sample(range(num_pts), min(batch_size, num_pts))
                p = points[sampled_ind]
                pred = self.decoder(p)
                sdf, deform = pred[:,0], pred[:,1:]
                loss = nn.functional.mse_loss(sdf, sdf_gt[sampled_ind])# + (deform ** 2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                mesh_v, mesh_f, _ = self.get_mesh(return_loss=False)
            pred_mesh = trimesh.Trimesh(mesh_v.cpu().numpy(), mesh_f.cpu().numpy())
            print('fitted mesh with num_vertex {}, num_faces {}'.format(mesh_v.shape[0], mesh_f.shape[0]))