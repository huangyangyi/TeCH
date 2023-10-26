import torch
from torch import Tensor, nn
import numpy as np


###############################################################################
# Marching tetrahedrons implementation (differentiable), adapted from
# https://github.com/NVIDIAGameWorks/kaolin/blob/master/kaolin/ops/conversions/tetmesh.py
###############################################################################
class DMTet(nn.Module):

    def __init__(self):
        super().__init__()
        triangle_table = torch.tensor([
            [-1, -1, -1, -1, -1, -1],
            [ 1,  0,  2, -1, -1, -1],
            [ 4,  0,  3, -1, -1, -1],
            [ 1,  4,  2,  1,  3,  4],
            [ 3,  1,  5, -1, -1, -1],
            [ 2,  3,  0,  2,  5,  3],
            [ 1,  4,  0,  1,  5,  4],
            [ 4,  2,  5, -1, -1, -1],
            [ 4,  5,  2, -1, -1, -1],
            [ 4,  1,  0,  4,  5,  1],
            [ 3,  2,  0,  3,  5,  2],
            [ 1,  3,  5, -1, -1, -1],
            [ 4,  1,  2,  4,  3,  1],
            [ 3,  0,  4, -1, -1, -1],
            [ 2,  0,  1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1],
        ], dtype=torch.long) # yapf: disable


        num_triangles_table = torch.tensor([0, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 0], dtype=torch.long)
        base_tet_edges = torch.tensor([0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3], dtype=torch.long)

        self.register_buffer('triangle_table', triangle_table, persistent=False)
        self.register_buffer('num_triangles_table', num_triangles_table, persistent=False)
        self.register_buffer('base_tet_edges', base_tet_edges, persistent=False)

    ###############################################################################
    # Utility functions
    ###############################################################################

    def sort_edges(self, edges_ex2):
        with torch.no_grad():
            order = (edges_ex2[:, 0] > edges_ex2[:, 1]).long()
            order = order.unsqueeze(dim=1)

            a = torch.gather(input=edges_ex2, index=order, dim=1)
            b = torch.gather(input=edges_ex2, index=1 - order, dim=1)

        return torch.stack([a, b], -1)

    def map_uv(self, faces, face_gidx, max_idx):
        N = int(np.ceil(np.sqrt((max_idx + 1) // 2)))
        tex_y, tex_x = torch.meshgrid(
            torch.linspace(0, 1 - (1 / N), N, dtype=torch.float32, device=face_gidx.device),
            torch.linspace(0, 1 - (1 / N), N, dtype=torch.float32, device=face_gidx.device),
            indexing='ij')

        pad = 0.9 / N

        uvs = torch.stack([tex_x, tex_y, tex_x + pad, tex_y, tex_x + pad, tex_y + pad, tex_x, tex_y + pad],
                          dim=-1).view(-1, 2)

        def _idx(tet_idx, N):
            x = tet_idx % N
            y = torch.div(tet_idx, N, rounding_mode='trunc')
            return y * N + x

        tet_idx = _idx(torch.div(face_gidx, 2, rounding_mode='trunc'), N)
        tri_idx = face_gidx % 2

        uv_idx = torch.stack((tet_idx * 4, tet_idx * 4 + tri_idx + 1, tet_idx * 4 + tri_idx + 2), dim=-1).view(-1, 3)

        return uvs, uv_idx

    ###############################################################################
    # Marching tets implementation
    ###############################################################################

    def __call__(self, pos_nx3: Tensor, sdf_n: Tensor, tet_fx4: Tensor, with_uv: bool=True):
        with torch.no_grad():
            occ_n = sdf_n > 0
            occ_fx4 = occ_n[tet_fx4.reshape(-1)].reshape(-1, 4)
            occ_sum = torch.sum(occ_fx4, -1)
            valid_tets = (occ_sum > 0) & (occ_sum < 4)
            occ_sum = occ_sum[valid_tets]

            # find all vertices
            all_edges = tet_fx4[valid_tets][:, self.base_tet_edges].reshape(-1, 2)
            all_edges = self.sort_edges(all_edges)
            unique_edges, idx_map = torch.unique(all_edges, dim=0, return_inverse=True)

            unique_edges = unique_edges.long()
            mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1, 2).sum(-1) == 1
            mapping = torch.ones((unique_edges.shape[0]), dtype=torch.long, device="cuda") * -1
            mapping[mask_edges] = torch.arange(mask_edges.sum(), dtype=torch.long, device="cuda")
            idx_map = mapping[idx_map]  # map edges to verts

            interp_v = unique_edges[mask_edges]
        edges_to_interp = pos_nx3[interp_v.reshape(-1)].reshape(-1, 2, 3)
        edges_to_interp_sdf = sdf_n[interp_v.reshape(-1)].reshape(-1, 2, 1)
        edges_to_interp_sdf[:, -1] *= -1

        denominator = edges_to_interp_sdf.sum(1, keepdim=True)

        edges_to_interp_sdf = torch.flip(edges_to_interp_sdf, [1]) / denominator
        verts = (edges_to_interp * edges_to_interp_sdf).sum(1)

        idx_map = idx_map.reshape(-1, 6)

        v_id = torch.pow(2, torch.arange(4, dtype=torch.long, device="cuda"))
        tetindex = (occ_fx4[valid_tets] * v_id.unsqueeze(0)).sum(-1)
        num_triangles = self.num_triangles_table[tetindex]

        # Generate triangle indices
        faces = torch.cat(
            (
                torch.gather(
                    input=idx_map[num_triangles == 1],
                    dim=1,
                    index=self.triangle_table[tetindex[num_triangles == 1]][:, :3]).reshape(-1, 3),
                torch.gather(
                    input=idx_map[num_triangles == 2],
                    dim=1,
                    index=self.triangle_table[tetindex[num_triangles == 2]][:, :6]).reshape(-1, 3),
            ),
            dim=0,
        )
        if not with_uv:
            return verts, faces

        # Get global face index (static, does not depend on topology)
        num_tets = tet_fx4.shape[0]
        tet_gidx = torch.arange(num_tets, dtype=torch.long, device=tet_fx4.device)[valid_tets]
        face_gidx = torch.cat(
            (tet_gidx[num_triangles == 1] * 2,
             torch.stack((tet_gidx[num_triangles == 2] * 2, tet_gidx[num_triangles == 2] * 2 + 1), dim=-1).view(-1)),
            dim=0,
        )
        uvs, uv_idx = self.map_uv(faces, face_gidx, num_tets * 2)

        return verts, faces, uvs, uv_idx
