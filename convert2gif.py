from PIL import Image, GifImagePlugin
import glob
import imageio
import cv2
import numpy as np
from tkinter import filedialog as fd
import os
from pathlib import Path


def name_check():
    for name in glob.glob('E:/unet_dataset/imgs/*'):
        masks_dir=Path('E:/unet_dataset/masks/*')
        mask_file = list(masks_dir.glob(name + '_mask' + '.*'))
        #maskname='.'.join(name.split('.')[:-1]) + '.gif'
        #mask_file=Image.open(maskname)
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'



def cvt2gif():
    for name in glob.glob('E:/unet_dataset/masks/*'):
        if name.endswith(".jpg"):
            img = Image.open(name)
            newname = '.'.join(name.split('.')[:-1]) + '.gif'
            print(newname)
            ret, img=cv2.threshold(np.array(img), 240, 255, cv2.THRESH_BINARY)
            nomask=np.zeros(img.shape)
            img=Image.fromarray(img)
            nomask=Image.fromarray(nomask)
            img=img.convert('RGBA')
            nomask=nomask.convert('RGBA')
            imageio.mimsave(newname, [img, nomask])
            os.remove(name)

def gif_format():
    for name in glob.glob('E:/unet_dataset/masks/*'):
        if name.endswith(".gif"):
            img=Image.open(name)
            print(name)
            img._is_animated=False
            img.disposal_method=0
            img.info.clear()
            dic={'version':b'GIF89a', 'background':1, 'duration':0}
            img.info=dic
            img.im=None
            GifImagePlugin.GifImageFile.save(img, name)


def jpg_to_gif():
    global im1

    # import the image from the folder
    import_filename = fd.askopenfilename()
    if import_filename.endswith(".jpg"):

        im1 = Image.open(import_filename)

        # after converting the image save to desired
        # location with the Extersion .png
        export_filename = fd.asksaveasfilename(defaultextension=".gif")
        im1.save(export_filename)

        # displaying the Messaging box with the Success
        messagebox.showinfo("success ", "your Image converted to GIF Format")
    else:

        # if Image select is not with the Format of .jpg
        # then display the Error
        Label_2 = Label(root, text="Error!", width=20,
                        fg="red", font=("bold", 15))
        Label_2.place(x=80, y=280)
        messagebox.showerror("Fail!!", "Something Went Wrong...")


#name_check()
cvt2gif()
gif_format()