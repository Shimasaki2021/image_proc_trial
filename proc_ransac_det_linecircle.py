import sys
import os
import copy
import math
import time
import shutil
import cv2
import numpy as np
from enum import IntEnum, auto
from typing import List,Dict,Tuple,Any
from typing_extensions import deprecated
from io import TextIOWrapper

X = 0
Y = 1

class FigType:
    class Def(IntEnum):
        FIGTYPE_CIRCLE_ = auto()
        FIGTYPE_LINE_   = auto()
        FIGTYPE_NONE_   = auto()

    def __init__(self):
        self.figtype_ = FigType.Def.FIGTYPE_CIRCLE_
        return

    def next(self):
        if self.figtype_ == FigType.Def.FIGTYPE_CIRCLE_:
            self.figtype_ = FigType.Def.FIGTYPE_LINE_
        elif self.figtype_ == FigType.Def.FIGTYPE_LINE_:
            self.figtype_ = FigType.Def.FIGTYPE_NONE_
        else:
            pass
        return

    def isNone(self) -> bool:
        return self.figtype_ == FigType.Def.FIGTYPE_NONE_

    def __str__(self) -> str:
        ret_str = ""

        if self.figtype_ == FigType.Def.FIGTYPE_CIRCLE_:
            ret_str = "CIRCLE"
        elif self.figtype_ == FigType.Def.FIGTYPE_LINE_:
            ret_str = "LINE"
        else:
            ret_str = "NONE"

        return ret_str

class Fig:
    def __init__(self, cfg:Dict[str,Any]):
        self.is_valid_ = False
        self.num_inlier_ = 0
        self.inlier_pixels_:np.ndarray = None

        self.dist_th_ = float(cfg["INLIER_DIST_TH"])
        self.min_inlier_th_ = int(cfg["INLIER_NUM_MIN_TH"])
        return
    
    def reset(self):
        self.is_valid_ = False
        self.num_inlier_ = 0
        self.inlier_pixels_:np.ndarray = None
        return

    def choiseRandomPixels(self, pixels:np.ndarray) -> np.ndarray:
        return None
    
    def isEnableCreate(self, px:np.ndarray) -> bool:
        return False
    
    def create(self, px:np.ndarray):
        return

    def densityFilter(self, density_th:float) -> int:
        return self.num_inlier_

    @deprecated("低速版")
    def countInlier(self, pixels:np.ndarray, dist_th:float) -> int:
        self.num_inlier_ = 0
        return self.num_inlier_

    def countInlier2(self, pixels:np.ndarray, dist_th:float) -> int:
        self.num_inlier_ = 0
        return self.num_inlier_

    def erasePixels(self, img:np.ndarray) -> np.ndarray:
        if (self.is_valid_ == True) and (self.inlier_pixels_ is not None):
            # inlier点を削除(0塗りつぶし)する

            # for px in self.inlier_pixels_:
            #     img[px[Y], px[X]] = 0
            img[self.inlier_pixels_[:, Y], self.inlier_pixels_[:, X]] = 0

        return img
    
    def draw(self, img:np.ndarray) -> np.ndarray:
        return img

class FigLine(Fig):

    def __init__(self, cfg:Dict[str,Any]):
        super().__init__(cfg)

        # ax + by + c = 0
        self.a_ = 0.0
        self.b_ = 0.0
        self.c_ = 0.0
        self.sqrt_a2_plus_b2_ = 0.0 # √a^2 + b^2
        self.inlier_bbox_:np.ndarray = None
        self.len_lineseg_ = 0.0
        
        self.inlier_dense_th_ = float(cfg["INLIER_LINE_DENSE_TH"])
        return

    def choiseRandomPixels(self, pixels:np.ndarray) -> np.ndarray:
        # pixelsの中からランダムに2点を選ぶ（重複禁止）
        return pixels[np.random.choice(len(pixels), 2, False)]

    def isEnableCreate(self, px:np.ndarray) -> bool:
        is_create = True

        (x0,y0) = px[0]
        (x1,y1) = px[1]

        if (x0 == x1) and (y0 == y1):
            # [入力2点が同じ] 直線作成不可
            is_create = False

        return is_create

    def create(self, px:np.ndarray):
        pxf = px.astype(float)
        (x0,y0) = pxf[0]
        (x1,y1) = pxf[1]

        # 直線のパラメータa,b,cを算出
        #   直線の方向ベクトル＝(x1​−x0​, y1​−y0​)
        #     → 直線の法線ベクトル＝(y0​−y1​,x1​−x0​)＝パラメータ(a,b)
        a = y0 - y1
        b = x1 - x0
        c = -(a * x0 + b * y0)

        sqrt_a2_plus_b2 = math.sqrt(a**2 + b**2)

        if sqrt_a2_plus_b2 > 1e-5:
            self.a_ = a
            self.b_ = b
            self.c_ = c
            self.sqrt_a2_plus_b2_ = sqrt_a2_plus_b2

            self.is_valid_ = True

        return

    def calcInlierBBox(self, pixels:np.ndarray) -> np.ndarray:
        # inlier点群の外接矩形を作成
        bbox_min = pixels.min(0)
        bbox_max = pixels.max(0)
        inlier_bbox = np.array([bbox_min[X], bbox_min[Y], bbox_max[X], bbox_max[Y]])

        return inlier_bbox

    def calcLenLineseg(self) -> int:
        # inlier点群で形成される線分長≒外接矩形の長辺　に近似
        bbox_w = self.inlier_bbox_[2] - self.inlier_bbox_[0]
        bbox_h = self.inlier_bbox_[3] - self.inlier_bbox_[1]
        len_lineseg = bbox_w if bbox_w > bbox_h else bbox_h

        return len_lineseg
    

    def densityFilter(self, density_th:float) -> int:
        # 点群密度がdensity_th未満の場合は無効化（num_inlier＝0）
        min_inlier_th = int(density_th * float(self.len_lineseg_))

        if self.num_inlier_ < min_inlier_th:
            self.num_inlier_ = 0

        return self.num_inlier_

    @deprecated("低速版")
    def countInlier(self, pixels:np.ndarray, dist_th:float) -> int:
        self.num_inlier_ = 0
        inlier_pixels = []

        if self.is_valid_ == True:
            for px in pixels:
                # 点と直線の距離 < 閾値 を満たす点の数をカウント
                dist = math.fabs(self.a_ * float(px[X]) + self.b_ * float(px[Y]) + self.c_) / self.sqrt_a2_plus_b2_

                if dist < dist_th:
                    self.num_inlier_ += 1
                    inlier_pixels.append(px)

            if self.num_inlier_ > self.min_inlier_th_:
                # inlier点群の外接矩形/線分長を算出(近似)
                self.inlier_bbox_ = self.calcInlierBBox(pixels)
                self.len_lineseg_ = self.calcLenLineseg()

                # 点群密度が閾値未満の場合は無効化（num_inlier＝0）
                self.num_inlier_ = self.densityFilter(self.inlier_dense_th_)

                if self.num_inlier_ > 0:
                    self.inlier_pixels_ = np.array(inlier_pixels)

            else:
                self.num_inlier_ = 0

        return self.num_inlier_

    def countInlier2(self, pixels:np.ndarray, dist_th:float) -> int:
        self.num_inlier_ = 0

        if self.is_valid_ == True:
            # 点と直線の距離 < 閾値 を満たす点の数をカウント
            dist = np.abs((self.a_ * pixels[:,X] + self.b_ * pixels[:,Y] + self.c_)) / self.sqrt_a2_plus_b2_
            mask = dist < dist_th

            self.num_inlier_ = np.count_nonzero(mask)

            if self.num_inlier_ > self.min_inlier_th_:
                # inlier点群の外接矩形/線分長(近似)を算出
                self.inlier_bbox_ = self.calcInlierBBox(pixels)
                self.len_lineseg_ = self.calcLenLineseg()

                # 点群密度が閾値未満の場合は無効化（num_inlier＝0）
                self.num_inlier_ = self.densityFilter(self.inlier_dense_th_)

                if self.num_inlier_ > 0:
                    self.inlier_pixels_ = copy.deepcopy(pixels[mask])

            else:
                self.num_inlier_ = 0

        return self.num_inlier_

    def calcIntersectBBox(self, bbox:np.ndarray) -> np.ndarray:
        # 直線と外接矩形の交点（上下左右）算出
        (bmin_x, bmin_y, bmax_x, bmax_y) = bbox

        inter_px = []

        # 上端(y=bmin_y)/下端(y=bmax_y)との交点
        if math.fabs(self.a_) > 1e-5:
            y = bmin_y
            x = -(self.b_ * y + self.c_) / self.a_
            if (bmin_x <= x) and (x <= bmax_x):
                inter_px.append([x,y])
            
            y = bmax_y
            x = -(self.b_ * y + self.c_) / self.a_
            if (bmin_x <= x) and (x <= bmax_x):
                inter_px.append([x,y])

        # 左端(x=bmin_x)/右端(x=bmax_x)との交点
        if math.fabs(self.b_) > 1e-5:
            x = bmin_x
            y = -(self.a_ * x + self.c_) / self.b_
            if (bmin_y <= y) and (y <= bmax_y):
                inter_px.append([x,y])
            
            x = bmax_x
            y = -(self.a_ * x + self.c_) / self.b_
            if (bmin_y <= y) and (y <= bmax_y):
                inter_px.append([x,y])

        return np.array(inter_px)

    def draw(self, img:np.ndarray) -> np.ndarray:
        COL = (0,255,255)
        ALPHA = 0.6

        # 直線と点群の外接矩形の交点を算出
        inter_px = self.calcIntersectBBox(self.inlier_bbox_)

        if len(inter_px) >= 2:
            # 直線描画
            img_draw_layer = copy.deepcopy(img)

            inter_px = inter_px.astype(int)
            cv2.line(img_draw_layer, 
                     (inter_px[0][X], inter_px[0][Y]), 
                     (inter_px[1][X], inter_px[1][Y]), COL, 2, cv2.LINE_AA)
            
            img = cv2.addWeighted(img_draw_layer, ALPHA, img, 1.0-ALPHA, 0)

        return img


class FigCircle(Fig):

    def __init__(self, cfg:Dict[str,Any]):
        super().__init__(cfg)

        # x^2 + y^2 + ax + by + c = 0
        self.a_ = 0.0
        self.b_ = 0.0
        self.c_ = 0.0

        # 中心center、半径r
        self.center_ = np.array([0,0])
        self.r_ = 0

        self.inlier_dense_th_ = float(cfg["INLIER_CIRCLE_DENSE_TH"])
        self.min_r_th_ = int(cfg["CIRCLE_MIN_R_TH"])
        return

    def choiseRandomPixels(self, pixels:np.ndarray) -> np.ndarray:
        # pixelsの中からランダムに3点を選ぶ（重複禁止）
        return pixels[np.random.choice(len(pixels), 3, False)]

    def isEnableCreate(self, px:np.ndarray) -> bool:
        is_create = True

        # 3点が一直線上にあるかどうかを判定
        #   → 3点で形成される三角形の面積が0かどうかで判定
        #   → 2ベクトルの外積が0かどうかで判定
        pxf = px.astype(float)
        (x0,y0) = pxf[0]
        (x1,y1) = pxf[1]
        (x2,y2) = pxf[2]

        cross = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)

        if abs(cross) < 1e-5:
            # [入力3点が一直線上に存在] 円作成不可
            is_create = False

        return is_create

    def create(self, px:np.ndarray):
        pxf = px.astype(float)
        (x0,y0) = pxf[0]
        (x1,y1) = pxf[1]
        (x2,y2) = pxf[2]

        # 円のパラメータa,b,cを算出
        #   → 連立方程式AP=Bを解く。P=[a,b,c]
        A = np.array([
            [x0, y0, 1.0],
            [x1, y1, 1.0],
            [x2, y2, 1.0]
        ])
        B = -np.array([
            x0**2 + y0**2,
            x1**2 + y1**2,
            x2**2 + y2**2
        ])
        (a, b, c) = np.linalg.solve(A, B)

        # a,b,cから円の中心、半径を算出
        cx = -a / 2.0
        cy = -b / 2.0
        r  = np.sqrt(cx**2 + cy**2 - c)

        if self.min_r_th_ < r:
            self.a_ = a
            self.b_ = b
            self.c_ = c
            self.center_[X] = int(cx)
            self.center_[Y] = int(cy)
            self.r_ = int(r)

            self.is_valid_ = True

        return
    
    def densityFilter(self, density_th:float) -> int:
        # 円周長を算出
        len_circle = 2.0 * math.pi * float(self.r_)

        # 点群密度がCIRCLE_INLIER_DENSE_TH未満の場合は無効化（num_inlier＝0）
        min_inlier_th = int(len_circle * density_th)
        # print(f"r={self.r_}, len_circle={len_circle}, min_inlier_th={min_inlier_th}, num_inlier={self.num_inlier_}")

        if self.num_inlier_ < min_inlier_th:
            self.num_inlier_ = 0

        return self.num_inlier_

    @deprecated("低速版")
    def countInlier(self, pixels:np.ndarray, dist_th:float) -> int:
        self.num_inlier_ = 0
        inlier_pixels = []

        if self.is_valid_ == True:
            for px in pixels:
                # 点と円周の距離 < 閾値 を満たす点の数をカウント
                #   点と円周の距離＝|点と円中心の距離 - 円半径|
                vec_px_center  = px - self.center_
                dist_px_center = math.sqrt(float(vec_px_center[X]**2 + vec_px_center[Y]**2))
                dist = math.fabs(dist_px_center - float(self.r_))

                if dist < dist_th:
                    self.num_inlier_ += 1
                    inlier_pixels.append(px)

            if self.num_inlier_ > self.min_inlier_th_:
                # 点群密度が閾値未満の場合は無効化（num_inlier＝0）
                self.num_inlier_ = self.densityFilter(self.inlier_dense_th_)

                if self.num_inlier_ > 0:
                    self.inlier_pixels_ = np.array(inlier_pixels)

            else:
                self.num_inlier_ = 0


        return self.num_inlier_

    def countInlier2(self, pixels:np.ndarray, dist_th:float) -> int:
        self.num_inlier_ = 0

        if self.is_valid_ == True:
            # 点と円周の距離 < 閾値 を満たす点の数をカウント
            #   点と円周の距離＝|点と円中心の距離 - 円半径|

            #   平方根計算を回避するため、判定式を以下にする
            #      (円半径 - 閾値)^2 < 点と円中心の距離^2 < (円半径 + 閾値)^2
            dist2 = np.sum((pixels -self.center_) ** 2, axis=1)

            r_min2 = (self.r_ - dist_th) ** 2
            r_max2 = (self.r_ + dist_th) ** 2

            mask = (r_min2 < dist2) & (dist2 < r_max2)

            self.num_inlier_  = np.count_nonzero(mask)

            if self.num_inlier_ > self.min_inlier_th_:
                # 点群密度が閾値未満の場合は無効化（num_inlier＝0）
                self.num_inlier_ = self.densityFilter(self.inlier_dense_th_)

                if self.num_inlier_ > 0:
                    self.inlier_pixels_ = copy.deepcopy(pixels[mask])

            else:
                self.num_inlier_ = 0

        return self.num_inlier_

    def draw(self, img:np.ndarray) -> np.ndarray:
        COL = (0,255,0)
        ALPHA = 0.6

        # 円描画
        img_draw_layer = copy.deepcopy(img)

        cv2.circle(img_draw_layer, (self.center_[X], self.center_[Y]), self.r_, COL, 2, cv2.LINE_AA)

        img = cv2.addWeighted(img_draw_layer, ALPHA, img, 1.0-ALPHA, 0)

        return img

class DebugOut:
    def __init__(self, outdir:str, fname_base:str):
        self.outdir_ = outdir
        self.fname_base_ = fname_base
        self.is_out_ = False

        self.log_fp_:TextIOWrapper = None
        return

    def createOutdir(self):
        if self.is_out_ == True:
            if os.path.isdir(self.outdir_) == True:
                shutil.rmtree(self.outdir_)

            os.makedirs(self.outdir_)
        return

    def dumpImg(self, img:np.ndarray, postfix:str):
        if self.is_out_ == True:
            cv2.imwrite(f"{self.outdir_}/{self.fname_base_}_{postfix}.png", img)
        return

    def openLogFile(self, fname:str):
        if (self.is_out_ == True) and (self.log_fp_ is None):
            self.createOutdir()
            self.log_fp_ = open(f"{self.outdir_}/{fname}", "w")
        return

    def closeLogFile(self):
        if (self.is_out_ == True) and (self.log_fp_ is not None):
            self.log_fp_.close()
            self.log_fp_ = None
        return

    def printLogLine(self, str_line:str):
        if (self.is_out_ == True) and (self.log_fp_ is not None):
            self.log_fp_.write(f"{str_line}\n")
            print(str_line)
        return

def extractObjectRANSAC(edge_pixels:np.ndarray, obj_type:FigType, cfg:Dict[str,Any]) -> Fig:

    if obj_type.figtype_ == FigType.Def.FIGTYPE_LINE_:
        target_fig = FigLine(cfg)
    else:
        target_fig = FigCircle(cfg)

    num_iter = int(float(len(edge_pixels)) * float(cfg["RANSAC_NUM_ITER_PER_EDGE"]))
    # print(f"num_iter = {num_iter}")

    num_max_inlier = 0
    best_fig = copy.deepcopy(target_fig)
    count_iter = 0

    while count_iter < num_iter:

        target_fig.reset()

        # エッジ点群から、直線／円の作成に必要な点（直線なら2点、円なら3点）をランダムに抽出
        choise_pixels = target_fig.choiseRandomPixels(edge_pixels)

        if target_fig.isEnableCreate(choise_pixels) == True:

            # 抽出した点から直線／円を作成
            target_fig.create(choise_pixels)

            # 作成した直線／円周上の点の数（inlier）をカウント
            # num_inlier = target_fig.countInlier(edge_pixels, target_fig.dist_th_)
            num_inlier = target_fig.countInlier2(edge_pixels, target_fig.dist_th_)

            if num_inlier > num_max_inlier:
                num_max_inlier = num_inlier
                best_fig = copy.deepcopy(target_fig)
    
            count_iter += 1

    # inlier数最大の直線／円を返す
    return best_fig


def extractObjects(img_edge:np.ndarray, dbg:DebugOut, cfg:Dict[str,Any]) -> List[Fig]:

    det_objs = []

    target_obj_type = FigType()

    while not target_obj_type.isNone():

        # エッジ画像からエッジ点群を抽出
        edge_pixels = cv2.findNonZero(img_edge)

        if len(edge_pixels) <= 0:
            break

        edge_pixels = edge_pixels.reshape(edge_pixels.shape[0], edge_pixels.shape[2]) # [n,1,2] → [n,2]
        # print(f"edge_pixels = {len(edge_pixels)}, {edge_pixels[0:3]}, {type(edge_pixels)}, {edge_pixels.shape}")

        # エッジ点群から直線／円を1つ検出
        det_obj = extractObjectRANSAC(edge_pixels, target_obj_type, cfg)

        if det_obj.is_valid_ == True:
            # [検出できた場合] 
            det_objs.append(det_obj)

            # 検出した直線／円に含まれるエッジ点(inlier点)を削除し、
            # 同じ種別の図形検出を継続
            img_edge = det_obj.erasePixels(img_edge)

            dbg.printLogLine(f"[{len(det_objs)}] detect {target_obj_type}")
            dbg.dumpImg(img_edge, f"edge_tmp{len(det_objs)}_{target_obj_type}")

        else:
            # [検出できなかった場合] 次の種別の検出図形へ
            target_obj_type.next()

    return det_objs


def extractEdge(img_in:np.ndarray, dbg:DebugOut) -> np.ndarray:
    # https://qiita.com/kotai2003/items/662c33c15915f2a8517e
    med_val = np.median(img_in)
    # sigma = 0.33
    sigma = np.std(img_in) / 255.0 # 画素値の標準偏差を0～1に正規化
    min_val = int(max(  0, (1.0 - sigma) * med_val))
    max_val = int(max(255, (1.0 + sigma) * med_val))
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
        "RANSAC_NUM_ITER_PER_EDGE" : 1.5,

        # 検出図形（直線or円）との距離閾値(inlier閾値)[pixel]
        "INLIER_DIST_TH" : 1.0, 

        # inlier点群の数の下限閾値[pixel]
        "INLIER_NUM_MIN_TH" : 10, 

        # inlier点群の密度(0～1)閾値
        "INLIER_LINE_DENSE_TH"   : 0.5, # 直線
        "INLIER_CIRCLE_DENSE_TH" : 0.5, # 円

        # 円の最小半径[pixel]
        "CIRCLE_MIN_R_TH" : 5,

        # 出力ディレクトリ
        "OUTPUT_DIR" : "output",
    }

    args = sys.argv

    if len(args) < 2:
        print("Usage: ", args[0], " [img file path]")
    else:
        main(args[1], cfg)

