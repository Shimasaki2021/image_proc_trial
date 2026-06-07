import os
import shutil
import cv2
import numpy as np
from io import TextIOWrapper

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
