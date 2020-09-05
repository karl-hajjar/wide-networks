import seaborn as sns

SNS_STYLES = ['darkgrid', 'whitegrid', 'dark', 'white', 'ticks']


def set_plot_options(style='darkgrid'):
    if style is not None:  # if style is None then don't set any style for seaborn
        if style == 'default':
            sns.set()
        elif style in SNS_STYLES:
            sns.set_style(style)
        else:
            raise ValueError("'style' argument must be one of {} but was {}".format(['default'] + SNS_STYLES, style))