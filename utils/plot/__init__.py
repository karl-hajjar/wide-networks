import matplotlib.pyplot as plt
import seaborn as sns

SNS_STYLES = ['darkgrid', 'whitegrid', 'dark', 'white', 'ticks']
DEFAULT_FONTS_MAP = {'axes': {'titlesize': 20, 'labelsize': 18},
                     'xtick': {'labelsize': 18},
                     'ytick': {'labelsize': 18},
                     'legend': {'fontsize': 16}}


def set_plot_options(style='darkgrid'):
    if style is not None:  # if style is None then don't set any style for seaborn
        if style == 'default':
            sns.set()
        elif style in SNS_STYLES:
            sns.set_style(style)
        else:
            raise ValueError("'style' argument must be one of {} but was {}".format(['default'] + SNS_STYLES, style))


def set_plot_fonts(fonts_map=None):
    """
    Sets fonts for different parts of the plots according to the arguments in the dict fonts_map.
    :param fonts_map: a dictionary mapping keys (e.g. 'axes', 'xtick', etc) to the parameter value pairs which need to
    be set.
    :return:
    """
    # see https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot
    if fonts_map is None:
        fonts_map = DEFAULT_FONTS_MAP
    for key, args in fonts_map.items():
        for k, v in args.items():
            plt.rc(key, **{k: v})
