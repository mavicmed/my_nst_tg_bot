import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mgimg
from IPython.display import HTML
matplotlib.rcParams['animation.embed_limit'] = 512

class Usr:
    def __init__(self,):
        self.image_folder = '349848500/40'


usr = Usr()

def create_gif(usr):


    fig = plt.figure(frameon=False, dpi=1)
    plt.axis('off')
    fig.set_size_inches(512, 512)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

    ims = []
    for i in range(5000,):
        try:
            image = mgimg.imread(f'{usr.image_folder}/result/{i}.png')
            im = plt.imshow(image, animated=True, aspect='auto')
        #     im.show()
            ims.append([im])
        except:
            pass
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ani = animation.ArtistAnimation(fig, ims, interval=350, blit=True, repeat_delay=800)
    # print('ani created')
    ani.save(f'{usr.image_folder}/result.gif', writer='imagemagick')
    return ani


if __name__ == '__main__':
    create_gif(usr)
