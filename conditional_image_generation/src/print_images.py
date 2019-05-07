from torchvision.utils import make_grid
import matplotlib.image

def print_grid(images, outfname, row_wise=True, plot_size=5):
    """
    Create a grid with all images.
    """

    print images.size()
    
    n_images, img_fm, img_sz,  _ = images.size()
    # if not row_wise:
    #     images = images.transpose(0, 1).contiguous()
    #images = images.view(n_images * n_columns, img_fm, img_sz, img_sz)
    images.add_(1).div_(2.0)

    N = 10
    grid = make_grid(images[:N].data.cpu(), nrow=( N))

    matplotlib.image.imsave(outfname, grid.numpy().transpose((1, 2, 0)))
