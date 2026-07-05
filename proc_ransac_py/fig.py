import copy
import math
import cv2
import numpy as np
from enum import IntEnum, auto
from typing import List,Dict,Tuple,Any,override
from typing_extensions import deprecated

X = 0
Y = 1

class FigType:
    class Def(IntEnum):
        FIGTYPE_CIRCLE_ = auto()
        FIGTYPE_LINE_   = auto()
        FIGTYPE_NONE_   = auto()

    def __init__(self, type=Def.FIGTYPE_CIRCLE_):
        self.figtype_ = type
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
    """モデル（直線 or 円）
    """

    def __init__(self, cfg:Dict[str,Any]):
        self.is_valid_   = False
        self.num_inlier_ = 0
        self.density_    = 0.0
        self.inlier_pixels_:np.ndarray = None
        self.inlier_bbox_:np.ndarray   = None

        self.dist_th_       = float(cfg["INLIER_DIST_TH"])
        self.min_inlier_th_ = int(cfg["INLIER_NUM_MIN_TH"])
        return
    
    def reset(self):
        self.is_valid_      = False
        self.num_inlier_    = 0
        self.density_       = 0.0
        self.inlier_pixels_ = None
        self.inlier_bbox_   = None
        return

    def choiseRandomPixels(self, pixels:np.ndarray) -> np.ndarray:
        return None
    
    def isEnableCreate(self, px:np.ndarray) -> bool:
        return False
    
    def create(self, px:np.ndarray):
        return

    def countInlier(self, pixels:np.ndarray, dist_th:float) -> int:
        self.num_inlier_    = 0
        self.density_       = 0.0
        self.inlier_pixels_ = None
        self.inlier_bbox_   = None
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
        """エッジ画像から、モデル周辺の点群(inlier)を削除
        Args:
            img (np.ndarray): [in] エッジ画像
        Returns:
            np.ndarray: [out] 削除後のエッジ画像
        """
        if (self.is_valid_ == True) and (self.inlier_pixels_ is not None):
            # inlier点を削除(0塗りつぶし)する

            # for px in self.inlier_pixels_:
            #     img[px[Y], px[X]] = 0
            img[self.inlier_pixels_[:, Y], self.inlier_pixels_[:, X]] = 0

        return img
    
    def draw(self, img:np.ndarray) -> np.ndarray:
        return img
    
    def __str__(self) -> str:
        val = f"[valid={self.is_valid_},num_inlier=,{self.num_inlier_},"
        return val

class FigLine(Fig):
    """直線モデル（ax+by+c=0）
    """

    def __init__(self, cfg:Dict[str,Any]):
        super().__init__(cfg)

        # ax + by + c = 0
        self.a_ = 0.0
        self.b_ = 0.0
        self.c_ = 0.0
        self.sqrt_a2_plus_b2_ = 0.0 # √a^2 + b^2

        self.len_lineseg_ = 0
        self.lineseg_pt_:np.ndarray = None
        
        self.inlier_dense_th_ = float(cfg["INLIER_LINE_DENSE_TH"])
        self.line_min_len_th_ = int(cfg["LINE_MIN_LEN_TH"])
        return

    @override
    def choiseRandomPixels(self, pixels:np.ndarray) -> np.ndarray:
        """直線作成に必要な点(2点)をランダムに抽出（重複禁止）
        Args:
            pixels (np.ndarray): [in] 点群 [[x0,y0][x1,y1],...]
        Returns:
            np.ndarray: [out] 直線作成に必要な点(2点) [[x0,y0][x1,y1]]
        """
        return pixels[np.random.choice(len(pixels), 2, False)]

    @override
    def isEnableCreate(self, px:np.ndarray) -> bool:
        """直線を作成可能かどうかを判定
        Args:
            px (np.ndarray): [in] 直線作成に必要な点(2点) [[x0,y0][x1,y1]]
        Returns:
            bool: [out] 判定結果(True:可能、False:不可能)
        """
        is_create = True

        (x0,y0) = px[0]
        (x1,y1) = px[1]

        if (x0 == x1) and (y0 == y1):
            # [入力2点が同じ] 直線作成不可
            is_create = False

        return is_create

    @override
    def create(self, px:np.ndarray):
        """直線作成 (ax+by+c=0)
        Args:
            px (np.ndarray): [in] 直線作成に必要な点(2点) [[x0,y0][x1,y1]]
        """
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

    @staticmethod
    def calcLenLineseg(lineseg_pt:np.ndarray) -> int:
        """線分長算出
        Args:
            lineseg_pt (np.ndarray): [in] 線分の両端点 [[x0,y0][x1,y1]]
        Returns:
            int: [out] 線分長
        """
        len_lineseg = 0

        if lineseg_pt is not None:
            # 線分の両端点の長さを算出
            vec         = lineseg_pt[1] - lineseg_pt[0]
            len_lineseg = int(math.sqrt(vec[0]*vec[0] + vec[1]*vec[1]))

        return len_lineseg

    @staticmethod
    def calcDensity(num_inlier:int, len_lineseg:int) -> float:
        """線分の密度算出
        Args:
            num_inlier (int):  [in] 線分を構成する点群の数(inlier)
            len_lineseg (int): [in] 線分長
        Returns:
            float: [out] 密度
        """
        density = 0.0
        if len_lineseg > 0:
            density = float(num_inlier) / float(len_lineseg)
        
        return density

    def extractLineSegPixels(self, inlier_pixels:np.ndarray, k=2.0) -> Tuple[np.ndarray, np.ndarray]:
        """線分を構成する点群、線分の両端点を抽出
        Args:
            pixels (np.ndarray): [in] モデル周辺の点群(inlier) [[x0,y0][x1,y1]...]
            k (float, optional): [in] 直線方向の標準偏差σの倍率. Defaults to 2.0.
        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                - [out] 線分を構成する点群 [[x0,y0][x1,y1]...]
                - [out] 線分の両端点 [[x0,y0][x1,y1]]
        """
        # -- 点群の重心mean、重心meanを原点とする各点の位置ベクトル --
        #   centered = [px[0]-mean, px[1]-mean,...]
        mean     = np.mean(inlier_pixels, axis=0)
        centered = inlier_pixels - mean

        # -- 直線の方向（主成分方向）の標準偏差sigma算出 --
        cov = np.cov(centered, rowvar=False, bias=True)
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # sigma = √ 固有値eigenvalues[0],eigenvalues[1]の大きい方
        sigma = np.sqrt(np.max(eigenvalues))
        # 直線の方向ベクトル（大きい方の固有値に対応する固有ベクトル）
        idx = np.argmax(eigenvalues)
        pc1 = eigenvectors[:, idx]

        # 直線方向ベクトル(pc1)への射影値 ＝ 重心から各点までの距離（符号付き）
        #   proj = [centered[0]・pc1, centered[1]・pc1, ...] 
        #     ※各点位置ベクトルcentered[]と直線方向ベクトルpc1の内積
        proj = centered @ pc1

        # -- k * sigma以内の点を、線分を構成する点群lineseg_pixelsとして抽出 --
        mask           = np.abs(proj) <= k * sigma
        lineseg_pixels = inlier_pixels[mask]

        # -- 両端点endpointsの抽出 --
        proj_filtered = proj[mask]
        min_idx   = np.argmin(proj_filtered)
        max_idx   = np.argmax(proj_filtered)
        endpoints = np.array([lineseg_pixels[min_idx], lineseg_pixels[max_idx]])

        return (lineseg_pixels, endpoints)

    @override
    def countInlier(self, pixels:np.ndarray, dist_th:float) -> int:
        """線分周辺の点(inlier)の数をカウント
        Args:
            pixels (np.ndarray): [in] 点群 [[x0,y0][x1,y1]...]
            dist_th (float):     [in] 距離閾値[pixel]
        Returns:
            int: [out] 線分周辺の点(inlier)の数
        Notes:
            以下も実行
            
            * 線分を構成する点群、線分の両端点を抽出
            * 密度算出（密度:inlier点数/線分長）
        """
        self.num_inlier_ = 0

        if self.is_valid_ == True:
            # 点群pixels[]と直線の距離算出
            #    dist[] = [dist0, dist1, ...]
            dist = np.abs((self.a_ * pixels[:,X] + self.b_ * pixels[:,Y] + self.c_)) / self.sqrt_a2_plus_b2_

            # 距離<閾値:true, 距離≧閾値:false
            #   mask[] = [true, false, ...]
            mask = dist < dist_th

            # inlier数カウント
            #   mask[]の中でtrue（≠0）の数をカウント
            self.num_inlier_ = np.count_nonzero(mask)

            if self.num_inlier_ > self.min_inlier_th_:
                # 線分を構成する点群、線分の両端点を抽出
                (self.inlier_pixels_, self.lineseg_pt_) = self.extractLineSegPixels(pixels[mask], 2.0)

                # 密度算出
                self.len_lineseg_ = self.calcLenLineseg(self.lineseg_pt_)
                self.density_     = self.calcDensity(self.num_inlier_, self.len_lineseg_)
            else:
                self.num_inlier_ = 0
                self.density_    = 0.0

            if self.num_inlier_ == 0:
                self.inlier_pixels_ = None
                self.lineseg_pt_    = None
                self.inlier_bbox_   = None

        return self.num_inlier_

    @override
    def filteredByInlierPixels(self) -> bool:
        """inlier点群の密度等で直線をフィルタリング（有効、無効判定）
        Returns:
            bool: [out] True:有効、False:無効
        """
        if (self.is_valid_ == True) and (self.num_inlier_ > 0):

            # 線分長が閾値未満の場合は無効化
            if self.len_lineseg_ < self.line_min_len_th_:
                self.is_valid_ = False
            else:
                # 点群密度が閾値未満の場合は無効化
                self.is_valid_ = self.density_ > self.inlier_dense_th_
        
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

    @override
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

    @override
    def __str__(self) -> str:
        val  = f"{super().__str__()},a={self.a_},b={self.b_},c={self.c_}"
        val += f",inlier_bbox=[(,{self.inlier_bbox_[0]},{self.inlier_bbox_[1]},)-"
        val += f"(,{self.inlier_bbox_[2]},{self.inlier_bbox_[3]},)],"
        val += f"len_lineseg=,{self.len_lineseg_},]"
        return val

class FigCircle(Fig):
    """円モデル（x^2 + y^2 + ax + by + c = 0）
    """

    def __init__(self, cfg:Dict[str,Any]):
        super().__init__(cfg)

        # x^2 + y^2 + ax + by + c = 0
        self.a_ = 0.0
        self.b_ = 0.0
        self.c_ = 0.0

        # 中心center、半径r
        self.center_ = np.array([0,0])
        self.r_      = 0

        self.inlier_dense_th_ = float(cfg["INLIER_CIRCLE_DENSE_TH"])
        self.min_r_th_        = int(cfg["CIRCLE_MIN_R_TH"])
        return

    @override
    def choiseRandomPixels(self, pixels:np.ndarray) -> np.ndarray:
        """円作成に必要な点(3点)をランダムに抽出（重複禁止）
        Args:
            pixels (np.ndarray): [in] 点群 [[x0,y0][x1,y1],...]
        Returns:
            np.ndarray: [out] 円作成に必要な点(3点) [[x0,y0][x1,y1][x2,y2]]
        """
        return pixels[np.random.choice(len(pixels), 3, False)]

    @override
    def isEnableCreate(self, px:np.ndarray) -> bool:
        """円を作成可能かどうかを判定
        Args:
            px (np.ndarray): [in] 円作成に必要な点(3点) [[x0,y0][x1,y1][x2,y2]]
        Returns:
            bool: [out] 判定結果(True:可能、False:不可能)
        """
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

    @override
    def create(self, px:np.ndarray):
        """円作成 (x^2 + y^2 + ax + by + c = 0)
        Args:
            px (np.ndarray): [in] 円作成に必要な点(3点) [[x0,y0][x1,y1][x2,y2]]
        """
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

    @staticmethod
    def calcDensity(num_inlier:int, r:float) -> float:
        """円の密度算出
        Args:
            num_inlier (int): [in] 円を構成する点群の数(inlier)
            r (float):        [in] 円の半径
        Returns:
            float: [out] 密度
        """
        density = 0.0
        if r > 0.0:
            len_arc = 2.0 * math.pi * r
            density = float(num_inlier) / len_arc

        return density

    @override
    def countInlier(self, pixels:np.ndarray, dist_th:float) -> int:
        """円周辺の点(inlier)の数をカウント
        Args:
            pixels (np.ndarray): [in] 点群 [[x0,y0][x1,y1]...]
            dist_th (float): [in] 距離閾値[pixel]
        Returns:
            int: [out] 円周辺の点(inlier)の数
        Notes:
            以下も実行
            
            * 密度算出（密度:inlier点数/円周長）
        """
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

                # 密度算出
                self.density_ = self.calcDensity(self.num_inlier_, float(self.r_))

            else:
                self.num_inlier_ = 0
                self.density_    = 0.0

            if self.num_inlier_ == 0:
                self.inlier_pixels_ = None
                self.inlier_bbox_   = None

        return self.num_inlier_

    @override
    def filteredByInlierPixels(self) -> bool:
        """inlier点群の密度等で円をフィルタリング（有効、無効判定）
        Returns:
            bool: [out] True:有効、False:無効
        """
        self.is_valid_ = self.density_ > self.inlier_dense_th_
        return self.is_valid_

    @override
    def erasePixels(self, img:np.ndarray) -> np.ndarray:
        """エッジ画像から、円周辺の点群(inlier)を削除
        Args:
            img (np.ndarray): [in] エッジ画像
        Returns:
            np.ndarray: [out] 削除後のエッジ画像
        """
        COL = (0,0,0)
        MARGIN = 2
        if (self.is_valid_ == True):

            # erase_thick = int(self.dist_th_) + MARGIN
            # cv2.circle(img, (self.center_[X], self.center_[Y]), self.r_, COL, erase_thick, cv2.LINE_4) # 円周のみ消去

            cv2.circle(img, (self.center_[X], self.center_[Y]), self.r_ + MARGIN, COL, cv2.FILLED, cv2.LINE_4) # 内部も消去

        return img

    @override
    def draw(self, img:np.ndarray) -> np.ndarray:
        COL = (0,255,0)
        ALPHA = 0.6

        # 円描画
        img_draw_layer = copy.deepcopy(img)

        cv2.circle(img_draw_layer, (self.center_[X], self.center_[Y]), self.r_, COL, 2, cv2.LINE_AA)

        img = cv2.addWeighted(img_draw_layer, ALPHA, img, 1.0-ALPHA, 0)

        return img

    @override
    def __str__(self) -> str:
        val  = f"{super().__str__()},a={self.a_},b={self.b_},c={self.c_}"
        val += f",center=({self.center_[X]},{self.center_[Y]})"
        val += f",r={self.r_}]"
        return val
