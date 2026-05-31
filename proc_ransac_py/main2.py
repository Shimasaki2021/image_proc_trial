import sys
import os
import copy
import time
import cv2
import numpy as np
from typing import List,Dict,Tuple,Any

from fig import FigType, Fig, FigLine, FigCircle
from debug import DebugOut

def nmSuppression(boxes:np.ndarray, scores:np.ndarray, iou_th:float, top_k:int=-1) -> Tuple[np.ndarray, int]:

    if len(boxes) == 0:
        return (np.array([], dtype=np.int64), 0)

    keep = np.zeros(len(scores), dtype=np.int64)
    count = 0

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1) * (y2 - y1)

    idx = np.argsort(scores)
    if top_k > 0:
        idx = idx[-top_k:]

    while idx.size > 0:
        i = idx[-1]
        keep[count] = i
        count += 1

        if idx.size == 1:
            break

        idx = idx[:-1]

        tmp_x1 = np.maximum(x1[idx], x1[i])
        tmp_y1 = np.maximum(y1[idx], y1[i])
        tmp_x2 = np.minimum(x2[idx], x2[i])
        tmp_y2 = np.minimum(y2[idx], y2[i])

        tmp_w = np.maximum(0.0, tmp_x2 - tmp_x1)
        tmp_h = np.maximum(0.0, tmp_y2 - tmp_y1)

        inter = tmp_w * tmp_h

        rem_areas = area[idx]
        union = rem_areas + area[i] - inter
        union = np.maximum(union, 1e-6)

        IoU = inter / union

        idx = idx[IoU <= iou_th]

    return (keep[:count], count)

def extractObjectRANSAC(edge_pixels:np.ndarray, obj_type:FigType, cfg:Dict[str,Any]) -> List[Fig]:

    det_objs:List[Fig] = []

    if obj_type.figtype_ == FigType.Def.FIGTYPE_LINE_:
        target_fig = FigLine(cfg)
        iou_th = 0.5
    else:
        target_fig = FigCircle(cfg)
        iou_th = 0.1

    num_iter = int(float(len(edge_pixels)) * float(cfg["RANSAC_NUM_ITER_PER_EDGE"]))

    count_iter = 0

    while count_iter < num_iter:

        target_fig.reset()

        # エッジ点群から、直線／円の作成に必要な点（直線なら2点、円なら3点）をランダムに抽出
        choise_pixels = target_fig.choiseRandomPixels(edge_pixels)

        if target_fig.isEnableCreate(choise_pixels) == True:

            # 抽出した点から直線／円を作成
            target_fig.create(choise_pixels)

            # 作成した直線／円周上の点の数（inlier）をカウント
            target_fig.countInlier(edge_pixels, target_fig.dist_th_)

            # inlier点群の特徴抽出（外接矩形）、フィルタリング
            target_fig.calcInlierBBox()
            is_valid = target_fig.filteredByInlierPixels()

            if is_valid == True:
                det_objs.append(copy.deepcopy(target_fig))
    
            count_iter += 1

    # 外接矩形が重複するものは、一番inlier数が多いもののみ残す（Non-maximum supression）
    boxes = [[det_obj.inlier_bbox_[0], det_obj.inlier_bbox_[1], det_obj.inlier_bbox_[2], det_obj.inlier_bbox_[3]] 
             for det_obj in det_objs]
    scores = [det_obj.num_inlier_ for det_obj in det_objs]

    (sup_idx , _) = nmSuppression(np.array(boxes), 
                                  np.array(scores),
                                  iou_th)

    det_objs_sup = [det_objs[i] for i in sup_idx]

    return det_objs_sup


def extractObjects(img_edge:np.ndarray, dbg:DebugOut, cfg:Dict[str,Any]) -> List[Fig]:

    det_objs_all = []

    target_obj_type = FigType()

    while not target_obj_type.isNone():

        # エッジ画像からエッジ点群を抽出
        edge_pixels = cv2.findNonZero(img_edge)

        if len(edge_pixels) <= 0:
            break

        edge_pixels = edge_pixels.reshape(edge_pixels.shape[0], edge_pixels.shape[2]) # [n,1,2] → [n,2]

        # エッジ点群から直線／円を検出
        det_objs = extractObjectRANSAC(edge_pixels, target_obj_type, cfg)

        if len(det_objs) > 0:
            # [検出できた場合] 
            det_objs_all += det_objs

            # 検出した直線／円に含まれるエッジ点(inlier点)を削除
            for det_obj in det_objs:
                img_edge = det_obj.erasePixels(img_edge)

            dbg.printLogLine(f"[{target_obj_type}] {len(det_objs)} detect.")
            dbg.dumpImg(img_edge, f"edge_tmp_after_{target_obj_type}")

        # 次の種別の検出図形へ
        target_obj_type.next()

    return det_objs_all


def extractEdge(img_in:np.ndarray, dbg:DebugOut) -> np.ndarray:
    # https://qiita.com/kotai2003/items/662c33c15915f2a8517e
    med_val = np.median(img_in)
    # sigma = 0.33
    sigma = np.std(img_in) / 255.0 # 画素値の標準偏差を0～1に正規化
    min_val = int(max(  0, (1.0 - sigma) * med_val))
    max_val = int(min(255, (1.0 + sigma) * med_val))
    img_edge = cv2.Canny(img_in, threshold1 = min_val, threshold2 = max_val)

    dbg.printLogLine(f"img_in.shape = {img_in.shape}")
    dbg.printLogLine(f"img_out({img_edge.shape} {img_edge.dtype}) = cv2.Canny(img_in, {min_val}, {max_val})")

    return img_edge


def main(img_fpath:str, cfg:Dict[str,Any]):

    img_in:np.ndarray = cv2.imread(img_fpath) 

    if img_in is not None:
        img_fname = os.path.basename(img_fpath)
        img_fname_base = os.path.splitext(img_fname)[0]

        dbg = DebugOut(cfg["OUTPUT_DIR"], img_fname_base)
        dbg.is_out_ = True
        dbg.openLogFile("log.txt")

        time_s = time.perf_counter()

        # エッジ検出
        img_in_g = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
        img_edge = extractEdge(img_in_g, dbg)
        dbg.dumpImg(img_edge, "edge")

        # 直線／円検出
        det_objs = extractObjects(img_edge, dbg, cfg)

        time_e = time.perf_counter()

        # 検出結果を重畳描画
        for det_obj in det_objs:
            img_in = det_obj.draw(img_in)

        dbg.dumpImg(img_in, "det")
        dbg.printLogLine(f"time[sec] = {time_e - time_s}")

        dbg.closeLogFile()

        cv2.imshow(img_fname, img_in)
        cv2.waitKey(0)

    return

if __name__ == "__main__":
    cfg = {
        # RANSAC繰り返し回数（エッジ点数に対する倍率を指定）
        # "RANSAC_NUM_ITER_PER_EDGE" : 1.5,
        "RANSAC_NUM_ITER_PER_EDGE" : 6.0,

        # 検出図形（直線or円）との距離閾値(inlier閾値)[pixel]
        "INLIER_DIST_TH" : 1.0, 

        # inlier点群の数の下限閾値[pixel]
        "INLIER_NUM_MIN_TH" : 10, 

        # inlier点群の密度(0～1)閾値
        "INLIER_LINE_DENSE_TH"   : 0.5, # 直線
        "INLIER_CIRCLE_DENSE_TH" : 0.5, # 円

        # 線分の最小長[pixel]
        "LINE_MIN_LEN_TH" : 150,
        # 円の最小半径[pixel]
        "CIRCLE_MIN_R_TH" : 5,

        # 出力ディレクトリ
        "OUTPUT_DIR" : "output_py2",
    }

    args = sys.argv

    if len(args) < 2:
        print("Usage: ", args[0], " [img file path]")
    else:
        main(args[1], cfg)

