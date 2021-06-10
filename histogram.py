import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import scale

from matplotlib.ticker import FuncFormatter

"""
for drawing the figure
"""


# # # def plot_hist(save_img_path, hist_array, bins, xmin, xmax, save_figure_name, figure_title):
def plot_hist(hist_array, bins, xmin, xmax, figure_title, file_addr, density=False, CDF=False):
    '''This function is used to plot the histogram, and the sum of it is 1'''
    # save_img_path : the path where you want to save the image
    # hist_array : the array that you want to plot its histogram
    # bins : how many bins do you want to plot
    # xmin, xmax : the minimun and the maximum of the x-axis
    # save_figure_name : the image file name
    # figure_title : the figure title

    plt.figure()
    n, bin, patches = plt.hist(hist_array, bins, (xmin, xmax), density=density)
    plt.close()
    plt.figure()
    plot_hist = n #/ bins
    fontsize = 20
    plt.title(figure_title, fontsize=fontsize)
    # manager = plt.get_current_fig_manager()
    # manager.resize(*manager.window.maxsize())
    x = np.arange(xmin, xmax, (xmax - xmin) / bins)
    # print(x, plot_hist)
    if CDF:
        plot_hist = np.cumsum(plot_hist * (xmax - xmin) / bins)
    plt.bar(x, plot_hist, align='center', width=(xmax - xmin) / bins)
    # print(file_addr)
    save_img_path = ('./'+ file_addr + '/Figure_' + figure_title+ '.png')
    plt.savefig(save_img_path)
    plt.title(figure_title)
    # plt.show()


def to_percent(temp, position):
    if temp == 1:
        return '%1.1f$\mathregular{T_w}$'%(temp)
    else:
        return '%1.1f'%(temp)

def plot_hist_multi(hist_array, bins, figure_title, file_addr, density=False, CDF=False, ndim=2, label=None, unit=False):
    '''This function is used to plot the histogram, and the sum of it is 1'''
    # save_img_path : the path where you want to save the image
    # hist_array : the array that you want to plot its histogram
    # bins : how many bins do you want to plot
    # xmin, xmax : the minimun and the maximum of the x-axis
    # save_figure_name : the image file name
    # figure_title : the figure title
    ndim = len(hist_array)
    n_bin = bins
    xmin = 1e10
    xmax = -1e10
    fontsize = 20
    plt.rcParams.update({'font.size': fontsize})
    plt.figure()
    if unit:
        plt.xlim(0,1)
    if CDF:
        plt.ylim(0,0.1)
    for i in hist_array:
        xmin = min(xmin, min(i))
        xmax = max(xmax, max(i))
    
    for i in range(ndim):
        # n, bin, patches = plt.hist(hist_array, bins, (xmin, xmax), density=density)
        list_X = hist_array[i]
        n, bin_edges = np.histogram(list_X, bins=n_bin)
        print(n)
        if density and not CDF:
            n = n / sum(n)
        print(n, sum(n))
        # plt.close()
        # plt.figure()
        plot_hist = n #/ bins
        
        
        # manager = plt.get_current_fig_manager()
        # manager.resize(*manager.window.maxsize())
        # print(x, plot_hist)
        if CDF:
            plot_hist = np.cumsum(plot_hist * (xmax - xmin) / n_bin)
        # plt.bar(x, plot_hist, align='center', width=(xmax - xmin) / n_bin, alpha=0.2)
        # bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
        bin_centers = bin_edges[:-1]
        if unit:
            x = np.arange(0, 1, 1 / n_bin)
            plt.plot(x, plot_hist, '-', label=label[i])
            
        else:
            plt.plot(bin_centers, plot_hist, '-', label=label[i])
        
    # if CDF:
    plt.ylim(0,0.1)
    plt.xlim(0,2) #DAWN congress-bills 20 threads-ask-ubuntu 2 tags 10
    
    # print(file_addr)
    fontsize = 20
    plt.legend()
    plt.title(figure_title, fontsize=fontsize)

    # plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    # plt.gca().xaxis.set_major_formatter(FuncFormatter(to_percent))
    # plt.xlabel('10% T')
    save_img_path = ('./'+ file_addr + '/Figure_' + figure_title+ '.png')
    plt.savefig(save_img_path)
    # plt.title(figure_title)
    # plt.show()

# def plot_hist_multi_bk(hist_array, bins, figure_title, file_addr, density=False, CDF=False, ndim=2, label=None, unit=False):
#     '''This function is used to plot the histogram, and the sum of it is 1'''
#     # save_img_path : the path where you want to save the image
#     # hist_array : the array that you want to plot its histogram
#     # bins : how many bins do you want to plot
#     # xmin, xmax : the minimun and the maximum of the x-axis
#     # save_figure_name : the image file name
#     # figure_title : the figure title
#     ndim = len(hist_array)
#     n_bin = bins
#     xmin = 1e10
#     xmax = -1e10
#     fontsize = 20
#     plt.rcParams.update({'font.size': fontsize})
#     plt.figure()
#     if unit:ls
#         plt.xlim(0,1)
#     if CDF:
#         plt.ylim(0,1)
#     for i in hist_array:
#         xmin = min(xmin, min(i))
#         xmax = max(xmax, max(i))
    
#     for i in range(ndim):
#         # n, bin, patches = plt.hist(hist_array, bins, (xmin, xmax), density=density)
#         list_X = hist_array[i]
#         n, bin_edges = np.histogram(list_X, bins=n_bin, normed=density)
#         # print(n)
#         # if density and not CDF:
#         #     n = n / sum(n)
#         # print(n, sum(n))
#         # plt.close()
#         # plt.figure()
#         plot_hist = n #/ bins
        
        
#         # manager = plt.get_current_fig_manager()
#         # manager.resize(*manager.window.maxsize())
#         # print(x, plot_hist)
#         if CDF:
#             plot_hist = np.cumsum(plot_hist * (xmax - xmin) / n_bin)
#         # plt.bar(x, plot_hist, align='center', width=(xmax - xmin) / n_bin, alpha=0.2)
#         # bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
#         bin_centers = bin_edges[:-1]
#         if unit:
#             x = np.arange(0, 1, 1 / n_bin)
#             plt.plot(x, plot_hist, '-', label=label[i])
            
#         else:
#             plt.plot(bin_centers, plot_hist, '-', label=label[i])
        
#     # print(file_addr)
#     fontsize = 20
#     plt.legend()
#     plt.title(figure_title, fontsize=fontsize)
#     save_img_path = ('./'+ file_addr + '/Figure_' + figure_title+ '.png')
#     plt.savefig(save_img_path)
#     # plt.title(figure_title)
#     # plt.show()