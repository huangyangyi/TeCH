import pyvista as pv
import pymeshlab
import tetgen
import os.path as osp
import os
import numpy as np

def build_tet_grid(mesh, cfg):
    assert cfg.data.last_model.split('.')[-1] == 'obj'
    tet_dir = osp.join(cfg.workspace, 'tet')
    os.makedirs(tet_dir, exist_ok=True)
    save_path = osp.join(tet_dir, 'tet_grid.npz')
    if osp.exists(save_path):
        print('Loading exist tet grids from {}'.format(save_path))
        tets = np.load(save_path)
        vertices = tets['vertices']
        indices = tets['indices']
        print('shape of vertices: {}, shape of grids: {}'.format(vertices.shape, indices.shape))
        return vertices, indices
    print('Building tet grids...')
    tet_flag = False
    tet_shell_offset = cfg.model.tet_shell_offset
    while (not tet_flag) and tet_shell_offset > cfg.model.tet_shell_offset / 16:
        # try:
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(mesh.v.cpu().numpy(), mesh.f.cpu().numpy()))
        ms.generate_resampled_uniform_mesh(offset=pymeshlab.AbsoluteValue(tet_shell_offset))
        ms.save_current_mesh(osp.join(tet_dir, 'dilated_mesh.obj'))
        mesh = pv.read(osp.join(tet_dir, 'dilated_mesh.obj'))
        downsampled_mesh = mesh.decimate(cfg.model.tet_shell_decimate)
        tet = tetgen.TetGen(downsampled_mesh)
        tet.make_manifold(verbose=True)
        vertices, indices = tet.tetrahedralize( fixedvolume=1, 
                                        maxvolume=cfg.model.tet_grid_volume, 
                                        regionattrib=1, 
                                        nobisect=False, steinerleft=-1, order=1, metric=1, meditview=1, nonodewritten=0, verbose=2)
        shell = tet.grid.extract_surface()
        shell.save(osp.join(tet_dir, 'shell_surface.ply'))
        np.savez(save_path, vertices=vertices, indices=indices)
        print('shape of vertices: {}, shape of grids: {}'.format(vertices.shape, indices.shape))
        tet_flag = True
        # except:
        #     tet_shell_offset /= 2
    assert tet_flag, "Failed to initialize tetrahedra grid!"
    return vertices, indices