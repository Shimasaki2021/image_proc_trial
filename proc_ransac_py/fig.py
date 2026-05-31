import copy
import math
import cv2
import numpy as np
from enum import IntEnum, auto
from typing import List,Dict,Tuple,Any
from typing_extensions import deprecated

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
        self.inlier_bbox_:np.ndarray = None

        self.dist_th_ = float(cfg["INLIER_DIST_TH"])
        self.min_inlier_th_ = int(cfg["INLIER_NUM_MIN_TH"])
        return
    
    def reset(self):
        self.is_valid_ = False
        self.num_inlier_ = 0
        self.inlier_pixels_ = None
        self.inlier_bbox_ = None
        return

    def choiseRandomPixels(self, pixels:np.ndarray) -> np.ndarray:
        return None
    
    def isEnableCreate(self, px:np.ndarray) -> bool:
        return False
    
    def create(self, px:np.ndarray):
        return

    def densityFilter(self, density_th:float) -> bool:
        return True

    @deprecated("低速版")
    def countInlier(self, pixels:np.ndarray, dist_th:float) -> int:
        self.num_inlier_ = 0
        return self.num_inlier_

    def countInlier2(self, pixels:np.ndarray, dist_th:float) -> int:
        self.num_inlier_ = 0
        self.inlier_pixels_ = None
        self.inlier_bbox_ = None
        return self.num_inlier_

    def calcInlierBBox(self):
        if self.num_inlier_ > 0:
            # inlier点群の外接矩形を作成
            bbox_min = self.inlier_pixels_.min(0)
            bbox_max = self.inlier_pixels_.max(0)
            self.inlier_bbox_ = np.array([bbox_min[X], bbox_min[Y], bbox_max[X], bbox_max[Y]])

        return

    def filteredByInlierPixels(self) -> bool:
        return False
        # return self.is_valid_

    def erasePixels(self, img:np.ndarray) -> np.ndarray:
        if (self.is_valid_ == True) and (self.inlier_pixels_ is not None):
            # inlier点を削除(0塗りつぶし)する

            # for px in self.inlier_pixels_:
            #     img[px[Y], px[X]] = 0
            img[self.inlier_pixels_[:, Y], self.inlier_pixels_[:, X]] = 0

        return img
    
    def draw(self, img:np.ndarray) -> np.ndarray:
        return img
    
    def __str__(self) -> str:
        val = f"[valid={self.is_valid_},num_inlier={self.num_inlier_},"
        return val

class FigLine(Fig):

    def __init__(self, cfg:Dict[str,Any]):
        super().__init__(cfg)

        # ax + by + c = 0
        self.a_ = 0.0
        self.b_ = 0.0
        self.c_ = 0.0
        self.sqrt_a2_plus_b2_ = 0.0 # √a^2 + b^2
        self.len_lineseg_ = 0
        
        self.inlier_dense_th_ = float(cfg["INLIER_LINE_DENSE_TH"])
        self.line_min_len_th_ = int(cfg["LINE_MIN_LEN_TH"])
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

    def calcLenLineseg(self) -> int:
        len_lineseg = 0

        if self.inlier_bbox_ is not None:
            # inlier点群で形成される線分長≒外接矩形の長辺　に近似
            bbox_w = self.inlier_bbox_[2] - self.inlier_bbox_[0]
            bbox_h = self.inlier_bbox_[3] - self.inlier_bbox_[1]
            len_lineseg = bbox_w if bbox_w > bbox_h else bbox_h

        return len_lineseg

    def densityFilter(self, density_th:float) -> bool:
        # 点群密度がdensity_th未満の場合は無効化
        min_inlier_th = int(density_th * float(self.len_lineseg_))

        if self.num_inlier_ < min_inlier_th:
            self.is_valid_ = False

        return self.is_valid_

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
                self.inlier_pixels_ = copy.deepcopy(pixels[mask])
            else:
                self.num_inlier_ = 0

            if self.num_inlier_ == 0:
                self.inlier_pixels_ = None
                self.inlier_bbox_ = None

        return self.num_inlier_

    def filteredByInlierPixels(self) -> bool:
        if (self.is_valid_ == True) and (self.num_inlier_ > 0):
            
            # inlier点群の外接矩形/線分長(近似)を算出
            self.len_lineseg_ = self.calcLenLineseg()

            # 線分長が閾値未満の場合は無効化
            if self.len_lineseg_ < self.line_min_len_th_:
                self.is_valid_ = False
            else:
                # 点群密度が閾値未満の場合は無効化
                self.is_valid_ = self.densityFilter(self.inlier_dense_th_)
        
        else:
            self.is_valid_ = False

        return self.is_valid_

    def calcIntersectBBox(self, bbox:np.ndarray) -> np.ndarray:
        # 直線と外接矩形の交点（上下左右）算出
        (bmin_x, bmin_y, bmax_x, bmax_y) = bbox

        inter_px = []

        # 上端(y=bmin_y)/下端(y=bmax_y)との交点
        if math.fabs(self.a_) > 1e-5:
            y = bmin_y
            x = -(self.b_ * y + self.c_) / self.a_
            if ((bmin_x - 1e-5) < x) and (x < (bmax_x + 1e-5)):
                inter_px.append([x,y])
            
            y = bmax_y
            x = -(self.b_ * y + self.c_) / self.a_
            if ((bmin_x - 1e-5) <= x) and (x < (bmax_x + 1e-5)):
                inter_px.append([x,y])

        # 左端(x=bmin_x)/右端(x=bmax_x)との交点
        if math.fabs(self.b_) > 1e-5:
            x = bmin_x
            y = -(self.a_ * x + self.c_) / self.b_
            if ((bmin_y - 1e-5) < y) and (y < (bmax_y + 1e-5)):
                inter_px.append([x,y])
            
            x = bmax_x
            y = -(self.a_ * x + self.c_) / self.b_
            if ((bmin_y - 1e-5) < y) and (y < (bmax_y + 1e-5)):
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

    def __str__(self) -> str:
        val  = f"{super().__str__()},a={self.a_},b={self.b_},c={self.c_}"
        val += f",inlier_bbox=[({self.inlier_bbox_[0]},{self.inlier_bbox_[1]})-"
        val += f"({self.inlier_bbox_[2]},{self.inlier_bbox_[3]})],"
        val += f"len_lineseg={self.len_lineseg_}]"
        return val

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
    
    def densityFilter(self, density_th:float) -> bool:
        # 円周長を算出
        len_circle = 2.0 * math.pi * float(self.r_)

        # 点群密度がCIRCLE_INLIER_DENSE_TH未満の場合は無効化（num_inlier＝0）
        min_inlier_th = int(len_circle * density_th)

        if self.num_inlier_ < min_inlier_th:
            self.is_valid_ = False

        return self.is_valid_

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
                self.inlier_pixels_ = copy.deepcopy(pixels[mask])
            else:
                self.num_inlier_ = 0

            if self.num_inlier_ == 0:
                self.inlier_pixels_ = None
                self.inlier_bbox_ = None

        return self.num_inlier_

    def filteredByInlierPixels(self) -> bool:
        self.is_valid_ = self.densityFilter(self.inlier_dense_th_)
        return self.is_valid_

    def erasePixels(self, img:np.ndarray) -> np.ndarray:
        COL = (0,0,0)
        MARGIN = 2
        if (self.is_valid_ == True):

            # erase_thick = int(self.dist_th_) + MARGIN
            # cv2.circle(img, (self.center_[X], self.center_[Y]), self.r_, COL, erase_thick, cv2.LINE_4) # 円周のみ消去

            cv2.circle(img, (self.center_[X], self.center_[Y]), self.r_ + MARGIN, COL, cv2.FILLED, cv2.LINE_4) # 内部も消去

        return img

    def draw(self, img:np.ndarray) -> np.ndarray:
        COL = (0,255,0)
        ALPHA = 0.6

        # 円描画
        img_draw_layer = copy.deepcopy(img)

        cv2.circle(img_draw_layer, (self.center_[X], self.center_[Y]), self.r_, COL, 2, cv2.LINE_AA)

        img = cv2.addWeighted(img_draw_layer, ALPHA, img, 1.0-ALPHA, 0)

        return img

    def __str__(self) -> str:
        val  = f"{super().__str__()},a={self.a_},b={self.b_},c={self.c_}"
        val += f",center=({self.center_[X]},{self.center_[Y]})"
        val += f",r={self.r_}]"
        return val
