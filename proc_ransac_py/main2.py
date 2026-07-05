import sys
import os
import copy
import time
import cv2
import numpy as np
from typing import List,Dict,Tuple,Any

from fig import FigType, Fig, FigLine, FigCircle
from debug import DebugOut
from main import extractEdge

def nmSuppression(boxes:np.ndarray, 
                  scores:np.ndarray, 
                  iou_th=0.45, 
                  top_k=-1) -> Tuple[np.ndarray, int]:
    """重複物体の削除
    Args:
        boxes (np.ndarray):       [in] 物体毎の外接矩形 [[xmin,ymin,xmax,ymax][xmin,ymin,xmax,ymax]..]
        scores (np.ndarray):      [in] 物体毎のscore（ここではinlier点数）
        iou_th (float, optional): [in] 外接矩形の重なり(IoU)閾値. Defaults to 0.45.
        top_k (int, optional):    [in] 削除後の物体数上限(-1:上限なし). Defaults to -1.

    Returns:
        Tuple[np.ndarray, int]: 
          - [out] 削除後の物体index [idx0,idx1,..]
          - [out] 削除後の物体数
    """

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
    if (top_k > 0) and (top_k < idx.shape[0]):
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

def extractObjectRANSAC(edge_pixels:np.ndarray, 
                        obj_type:FigType, 
                        cfg:Dict[str,Any]) -> List[Fig]:
    """エッジ点群から直線 or 円を複数検出(RANSAC)
    Args:
        edge_pixels (np.ndarray): [in] エッジ点群 [[x0,y0][x1,y1],...]
        obj_type (FigType):       [in] 検出するモデル種別（直線 or 円）
        cfg (Dict[str,Any]):      [in] config
    Returns:
        List[Fig]: [out] 検出結果（複数）（直線 or 円）
    """

    det_objs:List[Fig] = []

    if obj_type.figtype_ == FigType.Def.FIGTYPE_LINE_:
        target_fig = FigLine(cfg)
        iou_th = cfg["LINE_IOU_TH"]
    else:
        target_fig = FigCircle(cfg)
        iou_th = cfg["CIRCLE_IOU_TH"]

    num_iter = int(float(len(edge_pixels)) * float(cfg["RANSAC_NUM_ITER_PER_EDGE"]))

    count_iter = 0

    while count_iter < num_iter:

        target_fig.reset()

        # 観測データのサンプリング
        #   エッジ点群から、直線／円の作成に必要な点（直線なら2点、円なら3点）をランダムに抽出
        choise_pixels = target_fig.choiseRandomPixels(edge_pixels)

        if target_fig.isEnableCreate(choise_pixels) == True:

            # モデル作成（抽出した点から直線／円を作成）
            target_fig.create(choise_pixels)

            # モデル評価
            #   作成した直線／円周上の点の数（inlier）をカウント、密度算出
            target_fig.countInlier(edge_pixels, target_fig.dist_th_)

            # 外接矩形算出
            target_fig.calcInlierBBox()

            # 最良モデルの採用
            is_valid = target_fig.filteredByInlierPixels()

            if is_valid == True:
                det_objs.append(copy.deepcopy(target_fig))
    
            count_iter += 1

    # 重複物体の削除（Non-maximum supression）
    TOP_K = -1
    boxes = [[det_obj.inlier_bbox_[0], \
              det_obj.inlier_bbox_[1], \
              det_obj.inlier_bbox_[2], \
              det_obj.inlier_bbox_[3]] 
             for det_obj in det_objs]

    scores = [det_obj.num_inlier_ for det_obj in det_objs]

    (sup_idx , _) = nmSuppression(np.array(boxes), 
                                  np.array(scores),
                                  iou_th, 
                                  TOP_K)

    det_objs_sup = [det_objs[i] for i in sup_idx]

    return det_objs_sup


def extractObjects(img_edge:np.ndarray, 
                   dbg:DebugOut, 
                   cfg:Dict[str,Any]) -> List[Fig]:
    """複数の直線／円検出（複数まとめて検出）
    Args:
        img_edge (np.ndarray): [in] エッジ画像
        dbg (DebugOut):        [in] デバッグ
        cfg (Dict[str,Any]):   [in] config

    Returns:
        List[Fig]: [out] 検出結果（複数）（直線 or 円）
    """

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
            #   検出した直線／円のモデル周辺の点群を消去
            det_objs_all += det_objs

            for det_obj in det_objs:
                img_edge = det_obj.erasePixels(img_edge)

            dbg.printLogLine(f"[{target_obj_type}] {len(det_objs)} detect.")
            dbg.dumpImg(img_edge, f"edge_tmp_after_{target_obj_type}")

        # 次の種別の検出図形へ
        target_obj_type.next()

    return det_objs_all

def main(img_fpath:str, cfg:Dict[str,Any]):

    img_in:np.ndarray = cv2.imread(img_fpath) 

    if img_in is not None:
        # 乱数シード固定
        np.random.seed(cfg["RANDOM_SEED"])

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
        "RANSAC_NUM_ITER_PER_EDGE" : 4.0,

        # 検出図形（直線or円）との距離閾値(inlier閾値)[pixel]
        "INLIER_DIST_TH" : 1.0, 

        # inlier点群の数の下限閾値[pixel]
        "INLIER_NUM_MIN_TH" : 10, 

        # inlier点群の密度(0～1)閾値
        "INLIER_LINE_DENSE_TH"   : 0.5, # 直線
        "INLIER_CIRCLE_DENSE_TH" : 0.5, # 円

        # 線分の最小長[pixel]
        "LINE_MIN_LEN_TH" : 20,
        # 円の最小半径[pixel]
        "CIRCLE_MIN_R_TH" : 5,

        # IOU
        "LINE_IOU_TH": 0.10,
        "CIRCLE_IOU_TH": 0.05,

        # 出力ディレクトリ
        "OUTPUT_DIR" : "output_py2",

        # 乱数シード
        "RANDOM_SEED": 1000,
    }

    args = sys.argv

    if len(args) < 2:
        print("Usage: ", args[0], " [img file path]")
    else:
        main(args[1], cfg)

