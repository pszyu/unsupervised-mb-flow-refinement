import numpy as np
import cv2
import os
jp = os.path.join
import torch
import torch.nn.functional as F
from lib import flow_read, compute_unit_image_gradient
import argparse


def mb_by_flow_grad_bitemporal(flow_fw, flow_bw, thres=None):
    mag_fw = flow_grad_mag(flow_fw)
    mag_bw = flow_grad_mag(flow_bw)
    mag = (mag_fw + mag_bw) / 2.0
    if thres is None:
        return mag
    else:
        return mag > thres


def coords_grid(ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(1, 1, 1, 1).permute(0, 2, 3, 1)


def flow_grad_mag(flow, spacing=1):
    dx1, dy1 = np.gradient(flow[:, :, 0], spacing)
    dx2, dy2 = np.gradient(flow[:, :, 1], spacing)
    dx=np.sqrt(dx1**2 + dy1**2 + dx2**2 + dy2**2)
    return dx


def bilinear_sampler(img, coords, mode='bilinear'):
    """ Wrapper for grid_sample, uses pixel coordinates 
        Args:
            img: size [b, c, h, w]
            coords: size [b, h_r, w_r, 2]
        Returns:
            img: size [b, c, h_r, w_r]
    """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    new_img = F.grid_sample(img, grid)

    return new_img


def index_feature_window(input_fmap, coords, delta, r=4):
    '''
        Args:
            corr: size of [h, w, c]
            coords: [h, w, 2]
            r: radius
        Returns:
            indexed corr: size [h, w, c, (2r+1)**2]
    '''

    c1, h1, w1 = input_fmap.shape[1:]
    fmap = input_fmap.expand(h1*w1, -1, -1, -1)

    # delta size [2r+1, 2r+1, 2]
    centroid = coords.view(h1*w1, 1, 1, 2)    
    coords = centroid + delta

    fmap = bilinear_sampler(fmap, coords)
    # fmap has size [hw, c, 2r+1, 2r+1]
    fmap = fmap.view(h1, w1, c1, (2*r+1)**2).permute(3, 2, 0, 1)

    return fmap


def extract_color_patches(img, wsize):
    '''
        Args:
            img: [H, W, 3]
            wsize: The size of extracting window.
        return:
            cmap: [H, W, wsize**2]
    '''
    C, H, W = img.shape[1:]
    img = F.unfold(img, [wsize, wsize], padding=wsize//2, stride=1)
    # img has size [1, Cxwsizexwsize, HxW]
    img = img.view(1, -1, H, W)
    return img


def compute_mb_scores(fmap1, fmap2):
    '''
        Args:
            fmap1: [1, C, H, W]
            fmap2: [N, C, H, W]
        Returns:
            mb_scores: [N, H, W]
    '''
    norm_fmap1 = fmap1 - torch.mean(fmap1, axis=1, keepdims=True)
    norm_fmap2 = fmap2 - torch.mean(fmap2, axis=1, keepdims=True)
    
    mul12 = torch.sum(norm_fmap1 * norm_fmap2, axis=1)
    # mul12 has size [H, W, N]
    d1 = torch.sqrt(torch.sum(norm_fmap1**2, axis=1))
    # d1 [H, W, 1]
    d2 = torch.sqrt(torch.sum(norm_fmap2**2, axis=1))
    # d2 [H, W, N]
    d1d2 = d1 * d2 + 1e-6
    # d1d2 [H, W, N]
    mb_scores = mul12 / d1d2
    return mb_scores


def mb_score_by_twosides_change(m_bb, m_cb, m_bc, m_cc):
    mb_score, _ = torch.max(
        torch.stack([func_g(m_bb - m_cb), func_g(m_bc-m_cc)], axis=-1),
        axis=-1)
    return mb_score.cpu().numpy()


def func_g(x):
    return torch.abs(x) / 2


def twosides(img0, img1, img2, flow_bw, flow_fw, img_grad, 
             base_coords, delta, sigma=5, wsize=3, sr=1):
    b_loc = img_grad * sigma
    b_loc = b_loc
    c_loc = -b_loc

    cmap0 = extract_color_patches(img0, wsize)
    
    cmap1 = extract_color_patches(img1, wsize)
    cmap2 = extract_color_patches(img2, wsize)
    
    b_cmap1 = bilinear_sampler(cmap1, b_loc + base_coords)
    c_cmap1 = bilinear_sampler(cmap1, c_loc + base_coords)

    ms_bw = twosides_helper(b_cmap1, c_cmap1, cmap0, flow_bw, base_coords, b_loc, c_loc, delta, sr=sr)
    ms_fw = twosides_helper(b_cmap1, c_cmap1, cmap2, flow_fw, base_coords, b_loc, c_loc, delta, sr=sr)

    return ms_bw, ms_fw

    
def twosides_helper(b_cmap1, c_cmap1, cmap2, flow, base_coords, b_loc, c_loc, delta, sr):
    # b_cmap1/c_cmap1 have size [H, W, C, 1]

    b_flow = bilinear_sampler(flow, b_loc + base_coords).permute(0, 2, 3, 1)
    c_flow = bilinear_sampler(flow, c_loc + base_coords).permute(0, 2, 3, 1)
    
    bb_cmap2 = index_feature_window(cmap2, b_flow + base_coords + b_loc, delta, r=sr)
    cb_cmap2 = index_feature_window(cmap2, b_flow + base_coords + c_loc, delta, r=sr)
    bc_cmap2 = index_feature_window(cmap2, c_flow + base_coords + b_loc, delta, r=sr)
    cc_cmap2 = index_feature_window(cmap2, c_flow + base_coords + c_loc, delta, r=sr)
    
    ms_bb = compute_mb_scores(b_cmap1, bb_cmap2)
    ms_bc = compute_mb_scores(b_cmap1, bc_cmap2)
    ms_cb = compute_mb_scores(c_cmap1, cb_cmap2)
    ms_cc = compute_mb_scores(c_cmap1, cc_cmap2)

    m_bb, _ = torch.max(ms_bb, dim=0)
    m_bc, _ = torch.max(ms_bc, dim=0)
    m_cb, _ = torch.max(ms_cb, dim=0)
    m_cc, _ = torch.max(ms_cc, dim=0)

    return m_bb, m_cb, m_bc, m_cc


def get_boundary(p, img_size, search_range):
    h, w = p
    H, W = img_size
    ht, hb = max(0, h-search_range), min(h+search_range+1, H)
    wl, wr = max(0, w-search_range), min(w+search_range+1, W)
    return ht, hb, wl, wr


def flowgradmag_complement(mb_edge, mb_flowdiff, mb_twosides, search_range=2):
    H, W = mb_edge.shape
    mb_map = mb_flowdiff.copy()
    hs, ws = np.where(mb_flowdiff)
    mb_stack = [(hs[idx], ws[idx]) for idx in range(len(hs))]
    mb_candidates = mb_edge * mb_twosides
    
    while(len(mb_stack) > 0):
        p = mb_stack.pop()
        ht, hb, wl, wr = get_boundary(p, (H, W), search_range)
        for nh in range(ht, hb):
            for nw in range(wl, wr):
                if not mb_map[nh, nw] and mb_candidates[nh, nw]:
                    mb_map[nh, nw] = 1
                    mb_stack.append((nh, nw))
    return mb_map


def compute_delta(sr):
    delta_dx = torch.linspace(-sr, sr, 2*sr+1).cuda()
    delta_dy = torch.linspace(-sr, sr, 2*sr+1).cuda()
    delta = torch.stack(torch.meshgrid(delta_dy, delta_dx), dim=-1)
    delta = delta.view(1, 2*sr+1, 2*sr+1, 2)
    return delta


def pred_mb(img1, img2, img3, img2_grad, predflow_fw, predflow_bw, 
            gradmag_thres, base_coords=None, 
            sigma=5, wsize=3, sr=1, thres_twosides=0.1):
    # Computes delta, which can be shared by all the index_feature_window function.
    delta = compute_delta(sr)
    
    # Computes mb_flowdiff
    bflowdiff = mb_by_flow_grad_bitemporal(
        predflow_fw[0].permute(1, 2, 0).numpy(), predflow_bw[0].permute(1, 2, 0).numpy()) > gradmag_thres

    # Computes mb_twosides
    if base_coords is None:
        base_coords = coords_grid(img2_grad.shape[1], img2_arr.shape[2]).cuda()

    ms_bw, ms_fw = twosides(img1, img2, img3, predflow_bw.cuda(), predflow_fw.cuda(), img2_grad, base_coords, delta, sigma, wsize, sr)
    m_bb_bw, m_cb_bw, m_bc_bw, m_cc_bw = ms_bw
    m_bb_fw, m_cb_fw, m_bc_fw, m_cc_fw = ms_fw

    m_bb = torch.maximum(m_bb_fw, m_bb_bw)
    m_bc = torch.maximum(m_bc_fw, m_bc_bw)
    m_cb = torch.maximum(m_cb_fw, m_cb_bw)
    m_cc = torch.maximum(m_cc_fw, m_cc_bw)

    btwosides = mb_score_by_twosides_change(m_bb, m_cb, m_bc, m_cc) > thres_twosides
    pred_mb_twosides = flowgradmag_complement(bedge, bflowdiff, btwosides, search_range=2)
    
    return pred_mb_twosides


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unsupervised MB Detection and Flow Refinement')
    parser.add_argument('--gradmag_thres', type=int, default=1)
    parser.add_argument('--thres_twosides', type=float, default=0.1)
    parser.add_argument('--sigma', type=float, default=5)
    parser.add_argument('--wsize', type=float, default=3)
    parser.add_argument('--sr', type=float, default=1)
    parser.add_argument('--img1_path', type=str, default='')
    parser.add_argument('--img2_path', type=str, default='')
    parser.add_argument('--img3_path', type=str, default='')
    parser.add_argument('--edge2_path', type=str, default='')
    parser.add_argument('--flow23_path', type=str, default='')
    parser.add_argument('--flow21_path', type=str, default='')
    args = parser.parse_args()
    
    print('Is GPU Accessible?', torch.cuda.is_available())

    img2_arr = cv2.imread(args.img2_path)
    img2_grad = compute_unit_image_gradient(img2_arr)
    img2_grad = torch.Tensor(img2_grad)[None].cuda()
    img2 = torch.Tensor(img2_arr)[None].permute(0, 3, 1, 2).cuda()
    
    img1 = torch.Tensor(cv2.imread(args.img1_path))[None].permute(0, 3, 1, 2).cuda()
    img3 = torch.Tensor(cv2.imread(args.img3_path))[None].permute(0, 3, 1, 2).cuda()

    # Reads flow
    predflow_fw = torch.Tensor(flow_read(args.flow23_path))[None].permute(0, 3, 1, 2)
    predflow_bw = torch.Tensor(flow_read(args.flow21_path))[None].permute(0, 3, 1, 2)
    
    # Reads image edges
    bedge = cv2.imread(args.edge2_path, cv2.IMREAD_GRAYSCALE) < 120

    base_coords = coords_grid(img2_arr.shape[0], img2_arr.shape[1]).cuda()
    
    pred_mb_twosides = pred_mb(
        img1, img2, img3, img2_grad, predflow_fw, predflow_bw, 
        args.gradmag_thres, base_coords=base_coords, 
        sigma=args.sigma, wsize=args.wsize, sr=args.sr, 
        thres_twosides=args.thres_twosides)

    cv2.imwrite(
        jp('./demo', args.img2_path.split('/')[-1].replace('.png', '_predmb.png')), 
        pred_mb_twosides*255)