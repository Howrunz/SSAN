import os
import sys
import numpy as np
import cv2
import glob
import tempfile
from tqdm import tqdm
opj = os.path.join

class Latex:
    BASE = r'''
\documentclass[crop]{standalone}
\usepackage{ctex}
\usepackage{amsmath,fontspec,unicode-math,amssymb,mathptmx}
\usepackage[active,tightpage,displaymath,textmath]{preview}
\setmathfont{%s}
\begin{document}
\thispagestyle{empty}
\begin{equation}
%s
\nonumber
\end{equation}
\end{document}
'''

    def __init__(self, dpi=600, font='Fira Math'):
        self.dpi = dpi
        self.font = font
        self.prefix_line = self.BASE.split("\n").index("%s")

    def write(self, math, name, workdir, tardir, return_bytes=False):
        try:
            fd, texfile = tempfile.mkstemp('.tex', name, workdir, True)

            with os.fdopen(fd, 'w+') as f:
                document = self.BASE % (self.font, '\n'.join(math))
                f.write(document)

            self.convert_file(texfile, name, workdir, tardir, return_bytes=return_bytes)
        except Exception as e:
            print(e)
        finally:
            if os.path.exists(texfile):
                try:
                    os.remove(texfile)
                except PermissionError:
                    pass

    def convert_file(self, infile, name, workdir, tardir, return_bytes=False):
        infile = infile.replace('\\', '/')
        try:
            cmd = 'xelatex -interaction nonstopmode -file-line-error -output-directory %s %s >/dev/null 2>&1' % (workdir.replace('\\', '/'), infile)

            os.system(cmd)
            pdffile = infile.replace('.tex', '.pdf')
            jpgfile = os.path.join(tardir, name)

            cmd = 'convert -density %i -colorspace gray %s -quality 90 %s >/dev/null 2>&1' % (
                self.dpi,
                pdffile,
                jpgfile,
            )
            if sys.platform == 'win32':
                cmd = 'magick ' + cmd
            os.system(cmd)

        except Exception as e:
            print(e)
        finally:
            basefile = infile.replace('.tex', '')
            tempext = ['.aux', '.pdf', '.log']
            if return_bytes:
                ims = glob.glob(basefile+'*.jpg')
                for im in ims:
                    os.remove(im)
            for te in tempext:
                tempfile = basefile + te
                if os.path.exists(tempfile):
                    os.remove(tempfile)


def mergeLabel(label):
    char_list = label.split()
    char_list_space = [' ' + x + ' ' if '\\' in x else x for x in char_list]
    return ''.join(char_list_space)


def reconstruct_img(tardir, name, threshold=127):
    img_path = opj(tardir, name)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    threshold_map = np.min(img > threshold, axis=0).tolist()
    left_index = threshold_map.index(0)
    right_index = threshold_map[::-1].index(0)
    start_index = left_index - right_index if left_index - right_index >= 0 else right_index - left_index

    recons_img = img[:, start_index:]
    cv2.imwrite(opj(tardir, name), recons_img)


def formula2img(latex_list, workdir, tardir):
    latex2img = Latex()
    data_txt = open(opj(workdir, "true_crohme.txt"), "a", encoding="utf-8")
    data_mis_txt = open(opj(workdir, "false_crohme.txt"), "a", encoding="utf-8")

    with tqdm(total=len(latex_list), desc="Generate Formula", unit="imgs") as bar:
        for i, item in enumerate(latex_list):
            try:
                file_name, char_list = item.strip().split('\t')
            except:
                file_name = str(i)
                char_list = item.strip()
            str_latex = [mergeLabel(char_list)]
            bar.update(1)

            try:
                if not file_name.endswith(".jpg"):
                    file_name += ".jpg"
                latex2img.write(str_latex, file_name, workdir, tardir)
                reconstruct_img(tardir, file_name)
                data_txt.write(item)
            except Exception as e:
                print(e)
                data_mis_txt.write(item)
    data_txt.close()
    data_mis_txt.close()


def generate_print_img(label_path, target_path, work_path):
    with open(label_path, "r", encoding="utf-8") as f:
        data = f.readlines()
        formula2img(data, work_path, target_path)


if __name__ == "__main__":
    label_path = './data/19_test_labels.txt'
    target_path = './PrintedIMG'
    work_path = './temp'
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    if not os.path.exists(work_path):
        os.mkdir(work_path)

    generate_print_img(label_path, target_path, work_path)
