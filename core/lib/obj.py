import os
import cv2
import torch
import numpy as np


def dot(x, y):
    return torch.sum(x * y, -1, keepdim=True)


def length(x, eps=1e-20):
    return torch.sqrt(torch.clamp(dot(x, x), min=eps))


def safe_normalize(x, eps=1e-20):
    return x / length(x, eps)


def keep_largest(mesh):
    mesh_lst = mesh.split(only_watertight=False)
    keep_mesh = mesh_lst[0]
    for mesh in mesh_lst:
        if mesh.vertices.shape[0] > keep_mesh.vertices.shape[0]:
            keep_mesh = mesh
    return keep_mesh

def poisson(mesh, depth=10, face_count=500000):
    import open3d as o3d
    import trimesh
    pcd_path = "/tmp/_soups.ply"
    assert (mesh.vertex_normals.shape[1] == 3)
    mesh.export(pcd_path)
    pcl = o3d.io.read_point_cloud(pcd_path)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcl, depth=depth, n_threads=-1
        )

    mesh = trimesh.Trimesh(np.array(mesh.vertices), np.array(mesh.triangles))

    # only keep the largest component
    largest_mesh = keep_largest(mesh)

    return largest_mesh

class Mesh():

    def __init__(self, v=None, f=None, vn=None, fn=None, vt=None, ft=None, albedo=None, device=None, base=None, split=False):
        if split:
            import trimesh
            mesh = trimesh.Trimesh(v.cpu().detach().numpy(), f.cpu().detach().numpy(), process=True, validate=True)
            mesh = poisson(keep_largest(mesh))
            v = v.new_tensor(mesh.vertices)
            f = f.new_tensor(mesh.faces)
        self.v = v
        self.vn = vn
        self.vt = vt
        self.f = f
        self.fn = fn
        self.ft = ft
        self.v_color = None
        self.use_vertex_tex = False
        self.ref_v = None
        # only support a single albedo
        self.albedo = albedo
        self.device = device
        # copy non-None attribute from base
        if isinstance(base, Mesh):
            for name in ['v', 'vn', 'vt', 'f', 'fn', 'ft', 'albedo']:
                if getattr(self, name) is None:
                    setattr(self, name, getattr(base, name))

    # load from obj file
    @classmethod
    def load_obj(cls, path, albedo_path=None, device=None, init_empty_tex=False, use_vertex_tex=False, albedo_res=2048, ref_path=None, keypoints_path=None, init_uv=True):
        mesh = cls()

        # device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        if ref_path is not None:
            import trimesh
            mesh.ref_v = torch.tensor(trimesh.load(ref_path).vertices, dtype=torch.float32, device=device)
        else:
            mesh.ref_v = None


        assert os.path.splitext(path)[-1] == '.obj' or os.path.splitext(path)[-1] == '.ply'



        mesh.device = device

        # try to find texture from mtl file
        if albedo_path is None and '.obj' in path:
            mtl_path = path.replace('.obj', '.mtl')
            if os.path.exists(mtl_path):
                with open(mtl_path, 'r') as f:
                    lines = f.readlines()
                for line in lines:
                    split_line = line.split()
                    # empty line
                    if len(split_line) == 0:
                        continue
                    prefix = split_line[0]
                    # NOTE: simply use the first map_Kd as albedo!
                    if 'map_Kd' in prefix:
                        albedo_path = os.path.join(os.path.dirname(path), split_line[1])
                        print(f'[load_obj] use albedo from: {albedo_path}')
                        break

        if init_empty_tex or albedo_path is None or not os.path.exists(albedo_path):
            # init an empty texture
            print(f'[load_obj] init empty albedo!')
            # albedo = np.random.rand(1024, 1024, 3).astype(np.float32)
            albedo = np.ones((albedo_res, albedo_res, 3), dtype=np.float32) * np.array([0.5, 0.5, 0.5])  # default color
        else:
            albedo = cv2.imread(albedo_path, cv2.IMREAD_UNCHANGED)
            albedo = cv2.cvtColor(albedo, cv2.COLOR_BGR2RGB)
            albedo = albedo.astype(np.float32) / 255

            # import matplotlib.pyplot as plt
            # plt.imshow(albedo)
            # plt.show()

        mesh.albedo = torch.tensor(albedo, dtype=torch.float32, device=device)

        if os.path.splitext(path)[-1] == '.obj':

            # load obj
            with open(path, 'r') as f:
                lines = f.readlines()

            def parse_f_v(fv):
                # pass in a vertex term of a face, return {v, vt, vn} (-1 if not provided)
                # supported forms:
                # f v1 v2 v3
                # f v1/vt1 v2/vt2 v3/vt3
                # f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3
                # f v1//vn1 v2//vn2 v3//vn3
                xs = [int(x) - 1 if x != '' else -1 for x in fv.split('/')]
                xs.extend([-1] * (3 - len(xs)))
                return xs[0], xs[1], xs[2]

            # NOTE: we ignore usemtl, and assume the mesh ONLY uses one material (first in mtl)
            vertices, texcoords, normals = [], [], []
            faces, tfaces, nfaces = [], [], []
            for line in lines:
                split_line = line.split()
                # empty line
                if len(split_line) == 0:
                    continue
                # v/vn/vt
                prefix = split_line[0].lower()
                if prefix == 'v':
                    vertices.append([float(v) for v in split_line[1:]])
                elif prefix == 'vn':
                    normals.append([float(v) for v in split_line[1:]])
                elif prefix == 'vt':
                    val = [float(v) for v in split_line[1:]]
                    texcoords.append([val[0], 1.0 - val[1]])
                elif prefix == 'f':
                    vs = split_line[1:]
                    nv = len(vs)
                    v0, t0, n0 = parse_f_v(vs[0])
                    for i in range(nv - 2):  # triangulate (assume vertices are ordered)
                        v1, t1, n1 = parse_f_v(vs[i + 1])
                        v2, t2, n2 = parse_f_v(vs[i + 2])
                        faces.append([v0, v1, v2])
                        tfaces.append([t0, t1, t2])
                        nfaces.append([n0, n1, n2])
        elif os.path.splitext(path)[-1] == '.ply':
            vertices, texcoords, normals = [], [], []
            faces, tfaces, nfaces = [], [], []
            import trimesh
            trimesh_mesh = trimesh.load(path)
            vertices = trimesh_mesh.vertices
            faces = trimesh_mesh.faces
            if isinstance(trimesh_mesh.visual, trimesh.visual.ColorVisuals):
                vertices_colors = np.array(trimesh_mesh.visual.vertex_colors[:, :3]/255)
                vertices = np.concatenate([vertices, vertices_colors], axis=-1)
            

        mesh.v = torch.tensor(vertices, dtype=torch.float32, device=device)
        mesh.vt = torch.tensor(texcoords, dtype=torch.float32, device=device) if len(texcoords) > 0 else None
        mesh.vn = torch.tensor(normals, dtype=torch.float32, device=device) if len(normals) > 0 else None
        mesh.use_vertex_tex = use_vertex_tex
        if mesh.v.shape[1] == 6:
            mesh.v_color = mesh.v[:, 3:]
            mesh.v = mesh.v[:, :3]
        elif mesh.use_vertex_tex:
            mesh.v_color = torch.ones_like(mesh.v) * 0.5
        else:
            mesh.v_color = None

        mesh.f = torch.tensor(faces, dtype=torch.int32, device=device)
        mesh.ft = torch.tensor(tfaces, dtype=torch.int32, device=device) if texcoords is not None else None
        mesh.fn = torch.tensor(nfaces, dtype=torch.int32, device=device) if normals is not None else None

        if keypoints_path is not None:
            mesh.keypoints = np.load(keypoints_path, allow_pickle=True).item()['joints'].to(device)
            if len(mesh.keypoints.shape) == 2:
                mesh.keypoints = mesh.keypoints[None]
        elif len(mesh.v) == 6890: # SMPL mesh init
            import json
            with open('smpl_vert_segmentation.json') as f:
                segmentation = json.load(f)
                head_ind = segmentation['head']
            mesh.keypoints = mesh.v[head_ind].mean(dim=0)[None, None]
        elif mesh.ref_v is not None and len(mesh.ref_v) == 6890: # SMPL mesh init
            import json
            with open('smpl_vert_segmentation.json', 'r') as f:
                segmentation = json.load(f)
                head_ind = segmentation['head']
            mesh.keypoints = mesh.ref_v[head_ind].mean(dim=0)[None, None]
        else:
            mesh.keypoints = None
        print('mesh keypoints', mesh.keypoints.shape)

        # auto-normalize
        mesh.auto_size()

        print(f'[load_obj] v: {mesh.v.shape}, f: {mesh.f.shape}')

        # auto-fix normal
        if mesh.vn is None:
            mesh.auto_normal()

        print(f'[load_obj] vn: {mesh.vn.shape}, fn: {mesh.fn.shape}')

        # auto-fix texture
        if mesh.vt is None and not use_vertex_tex and init_uv:
            mesh.auto_uv(cache_path=path)

            print(f'[load_obj] vt: {mesh.vt.shape}, ft: {mesh.ft.shape}')

        return mesh

    # aabb
    def aabb(self):
        if hasattr(self, 'ref_v') and self.ref_v is not None:
            return torch.min(self.ref_v, dim=0).values, torch.max(self.ref_v, dim=0).values
        return torch.min(self.v, dim=0).values, torch.max(self.v, dim=0).values

    # unit size
    @torch.no_grad()
    def auto_size(self):  # to [-0.5, 0.5]
        vmin, vmax = self.aabb()
        scale = 1 / torch.max(vmax - vmin).item()
        self.v = self.v - (vmax + vmin) / 2  # Center mesh on origin
        v_c = (vmax + vmin) / 2
        self.v = self.v * scale
        if hasattr(self, 'keypoints') and self.keypoints is not None:
            self.keypoints = (self.keypoints - (vmax + vmin) / 2)*scale
        if hasattr(self, 'ref_v') and self.ref_v is not None:
            self.ref_v = (self.ref_v - (vmax + vmin) / 2)*scale
        self.resize_matrix_inv = torch.tensor([
            [1/scale, 0, 0, v_c[0]],
            [0, 1/scale, 0, v_c[1]],
            [0, 0, 1/scale, v_c[2]],
            [0, 0, 0, 1],
        ], dtype=torch.float, device=self.device)

    def auto_normal(self):
        i0, i1, i2 = self.f[:, 0].long(), self.f[:, 1].long(), self.f[:, 2].long()
        v0, v1, v2 = self.v[i0, :], self.v[i1, :], self.v[i2, :]
        face_normals = torch.cross(v1 - v0, v2 - v0)

        # Splat face normals to vertices
        vn = torch.zeros_like(self.v)
        vn.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
        vn.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
        vn.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)

        # Normalize, replace zero (degenerated) normals with some default value
        vn = torch.where(dot(vn, vn) > 1e-20, vn, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=vn.device))
        vn = safe_normalize(vn)
        #print('self.v.grad: {} face_normals: {} vn: {}'.format(self.v.requires_grad, face_normals.requires_grad, vn.requires_grad))

        self.vn = vn
        self.fn = self.f

    def auto_uv(self, cache_path=None):
        print('[INFO] Using atlas to calculate UV. It takes 10~20min.')
        # try to load cache
        if cache_path is not None:
            cache_path = cache_path.replace('.obj', '_uv.npz')
        if cache_path and os.path.exists(cache_path):
            data = np.load(cache_path)
            vt_np, ft_np = data['vt'], data['ft']
        else:

            import xatlas
            v_np = self.v.cpu().numpy() * 100
            f_np = self.f.int().cpu().numpy()
            atlas = xatlas.Atlas()
            atlas.add_mesh(v_np, f_np)
            chart_options = xatlas.ChartOptions()
            chart_options.max_iterations = 4
            atlas.generate(chart_options=chart_options)
            vmapping, ft_np, vt_np = atlas[0]  # [N], [M, 3], [N, 2]

            # save to cache
            if cache_path:
                np.savez(cache_path, vt=vt_np, ft=ft_np)

        vt = torch.from_numpy(vt_np.astype(np.float32)).to(self.device)
        ft = torch.from_numpy(ft_np.astype(np.int32)).to(self.device)

        self.vt = vt
        self.ft = ft

    def to(self, device):
        self.device = device
        for name in ['v', 'f', 'vn', 'fn', 'vt', 'ft', 'albedo']:
            tensor = getattr(self, name)
            if tensor is not None:
                setattr(self, name, tensor.to(device))
        return self

    # write to obj file
    def write(self, path):

        mtl_path = path.replace('.obj', '.mtl')
        albedo_path = path.replace('.obj', '_albedo.png')
        v_np = self.v.cpu().numpy()
        vt_np = self.vt.cpu().numpy() if self.vt is not None else None
        vn_np = self.vn.cpu().numpy() if self.vn is not None else None
        f_np = self.f.cpu().numpy()
        ft_np = self.ft.cpu().numpy() if self.ft is not None else None
        fn_np = self.fn.cpu().numpy() if self.fn is not None else None
        vc_np = self.v_color.cpu().numpy() if self.v_color is not None else None 
        print(f'vertice num: {len(v_np)}, face num: {len(f_np)}')

        with open(path, "w") as fp:
            fp.write(f'mtllib {os.path.basename(mtl_path)} \n')
            if self.use_vertex_tex:
                for v, c in zip(v_np, vc_np):
                    fp.write(f'v {v[0]} {v[1]} {v[2]} {c[0]} {c[1]} {c[2]}\n')
            else:
                for v in v_np:
                    fp.write(f'v {v[0]} {v[1]} {v[2]} \n')
                if vt_np is not None:
                    for v in vt_np:
                        fp.write(f'vt {v[0]} {1 - v[1]} \n')
                if vn_np is not None:
                    for v in vn_np:
                        fp.write(f'vn {v[0]} {v[1]} {v[2]} \n')
            if vt_np is not None:
                fp.write(f'usemtl defaultMat \n')
                for i in range(len(f_np)):
                    fp.write(
                        f'f {f_np[i, 0] + 1}/{ft_np[i, 0] + 1 if ft_np is not None else ""}/{fn_np[i, 0] + 1 if fn_np is not None else ""} \
                                {f_np[i, 1] + 1}/{ft_np[i, 1] + 1 if ft_np is not None else ""}/{fn_np[i, 1] + 1 if fn_np is not None else ""} \
                                {f_np[i, 2] + 1}/{ft_np[i, 2] + 1 if ft_np is not None else ""}/{fn_np[i, 2] + 1 if fn_np is not None else ""} \n'
                    )
            else:
                for i in range(len(f_np)):
                    fp.write(
                        f'f {f_np[i, 0] + 1} \
                                {f_np[i, 1] + 1} \
                                {f_np[i, 2] + 1} \n'
                    )


        if vt_np is not None:
            with open(mtl_path, "w") as fp:
                fp.write(f'newmtl defaultMat \n')
                fp.write(f'Ka 1 1 1 \n')
                fp.write(f'Kd 1 1 1 \n')
                fp.write(f'Ks 0 0 0 \n')
                fp.write(f'Tr 1 \n')
                fp.write(f'illum 1 \n')
                fp.write(f'Ns 0 \n')
                if not self.use_vertex_tex:
                    fp.write(f'map_Kd {os.path.basename(albedo_path)} \n')

            albedo = self.albedo.cpu().numpy()
            albedo = (albedo * 255).astype(np.uint8)
            cv2.imwrite(albedo_path, cv2.cvtColor(albedo, cv2.COLOR_RGB2BGR))
