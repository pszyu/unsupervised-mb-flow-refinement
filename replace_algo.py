import numpy as np
import os
import cv2
import argparse
from scipy import ndimage
from lib import flow_read, flow_write, compute_unit_image_gradient
jp = os.path.join


def single_epe(gtflow, predflow):
    return np.linalg.norm(gtflow-predflow, ord=2)


def replace(input_flow, point_list, replaced_flow, replaced_pmap, qmap_result):
    point_q = point_list[-1]
    flow_q = input_flow[point_q[1], point_q[0]]
    for point_p in point_list[:-1]:
        if not replaced_pmap[point_p[1], point_p[0]]:
            replaced_flow[point_p[1], point_p[0]] = flow_q
            replaced_pmap[point_p[1], point_p[0]] = 1
            qmap_result[point_p[1], point_p[0]] = point_q


def replace_flow(img_grad, input_flow, input_mb, 
                    q_dis=10, twosides_flowmag_diff_ratio=0.5, 
                    oneside_flow_diff_ratio=0.5):

    mbdis_map = ndimage.distance_transform_edt(1-input_mb)
    
    replaced_flow = np.copy(input_flow)
    replaced_pmap  = np.zeros(np.shape(mbdis_map))
    qmap_result = np.zeros_like(input_flow)
    
    H, W = input_mb.shape
    mb_hs, mb_ws = np.where(input_mb)
    
    for mb_idx, (mbh, mbw) in enumerate(zip(mb_hs, mb_ws)):
        # RepAlg on positive end point
        cloc = [mbw, mbh]
        pos_point_list = []
        
        first_ploc = np.array([mbw, mbh]) + img_grad[mbh, mbw] * 2
        first_ploc = [int(first_ploc[0]), int(first_ploc[1])]
        first_ploc_flow = input_flow[first_ploc[1], first_ploc[0]]
        curr_flow = first_ploc_flow
        pos_point_list.append(first_ploc)
        pos_idx = 2
        
        xp, yp = first_ploc
        if not (xp < 0 or xp >= W or yp < 0 or yp >= H):
            for idx in range(3, q_dis+1):
                ploc = np.array([mbw, mbh]) + img_grad[mbh, mbw] * idx
                xp, yp = int(ploc[0]), int(ploc[1])
                cloc = [xp, yp]
                pos_point_list.append(cloc)
                # 2 is buffer
                if (xp < 0 or xp >= W or yp < 0 or yp >= H or 
                    (idx > 2 and input_mb[yp, xp]) or 
                    mbdis_map[yp, xp] < idx - 2):
                    break
                else:
                    next_flow = input_flow[yp, xp]
                    curr_first_flow_dis = single_epe(curr_flow, first_ploc_flow)
                    curr_next_flow_dis = single_epe(curr_flow, next_flow)
                    if curr_next_flow_dis < oneside_flow_diff_ratio * curr_first_flow_dis:
                        pos_idx = idx - 1
                        break
                    curr_flow = next_flow

        # RepAlg on negative end point
        cloc = [mbw, mbh]
        neg_point_list = []
        
        first_nloc = np.array([mbw, mbh]) - img_grad[mbh, mbw] * 2
        first_nloc = [int(first_nloc[0]), int(first_nloc[1])]
        first_nloc_flow = input_flow[first_nloc[1], first_nloc[0]]
        curr_flow = first_nloc_flow
        neg_point_list.append(first_nloc)
        neg_idx = 2
        
        xp, yp = first_nloc
        if not (xp < 0 or xp >= W or yp < 0 or yp >= H):
            for idx in range(3, q_dis+1):
                nloc = np.array([mbw, mbh]) - img_grad[mbh, mbw] * idx
                xp, yp = int(nloc[0]), int(nloc[1])
                cloc = [xp, yp]
                neg_point_list.append(cloc)
                # 2 is buffer
                if xp < 0 or xp >= W or yp < 0 or yp >= H or (idx > 2 and input_mb[yp, xp]) or mbdis_map[yp, xp] < idx - 2:
                    break
                else:
                    next_flow = input_flow[yp, xp]
                    curr_first_flow_dis = single_epe(curr_flow, first_nloc_flow)
                    curr_next_flow_dis = single_epe(curr_flow, next_flow)
                    if curr_next_flow_dis < oneside_flow_diff_ratio * curr_first_flow_dis:
                        neg_idx = idx - 1
                        break
                    curr_flow = next_flow

        if pos_idx > 2 and neg_idx > 2:
            neg_point_q, pos_point_q = neg_point_list[-2], pos_point_list[-2]
            neg_flow_q = input_flow[neg_point_q[1], neg_point_q[0]]
            pos_flow_q = input_flow[pos_point_q[1], pos_point_q[0]]
            neg_flow_q_mag = single_epe(neg_flow_q, 0)
            pos_flow_q_mag = single_epe(pos_flow_q, 0)
            if single_epe(neg_flow_q, pos_flow_q) > twosides_flowmag_diff_ratio * min(neg_flow_q_mag, pos_flow_q_mag):
                if neg_flow_q_mag < pos_flow_q_mag:
                    selected_point_list = neg_point_list[:-1]
                else:
                    selected_point_list = pos_point_list[:-1]

                replace(input_flow, selected_point_list, replaced_flow, replaced_pmap, qmap_result)                

    return replaced_flow, replaced_pmap, qmap_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analysis of Replace Flow')
    parser.add_argument('--edge_path', type=str)
    parser.add_argument('--img_path', type=str)
    parser.add_argument('--input_mb_path', type=str)
    parser.add_argument('--input_flow_path', type=str)
    # Hyper-parameters
    # Common hyper-parameters
    parser.add_argument('--hp_q_dis', default=10, type=int)
    parser.add_argument('--twosides_flowmag_diff_ratio', default=0.2, type=float)
    parser.add_argument('--oneside_flow_diff_ratio', default=0.2, type=float)
    args = parser.parse_args()
    
    # Computes image gradient
    img1 = cv2.imread(args.img_path)
    img_grad = compute_unit_image_gradient(img1)

    input_mb = cv2.imread(args.input_mb_path, cv2.IMREAD_GRAYSCALE) > 120
    input_flow = flow_read(args.input_flow_path)
    # Thins the predicted MB via canny edges
    if args.edge_path is not None:
        edge = cv2.imread(args.edge_path, cv2.IMREAD_GRAYSCALE) > 120
    else:
        edge = cv2.Canny(img1, 100, 200)
    input_mb = input_mb * edge

    replaced_flow, replaced_pmap, qmap_result = replace_flow(
        img_grad=img_grad, input_flow=input_flow, 
        input_mb=input_mb, q_dis=args.hp_q_dis, 
        twosides_flowmag_diff_ratio=args.twosides_flowmag_diff_ratio,
        oneside_flow_diff_ratio=args.oneside_flow_diff_ratio)

    flow_file = args.input_flow_path.split('/')[-1]
    flow_write(replaced_flow, jp('./demo', flow_file.replace('.flo', '_replaced.flo')))
    np.save(jp('./demo', flow_file.replace('.flo', '_replaced_pmap.npy')), replaced_pmap)