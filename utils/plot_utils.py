def minimal_tick_params():
    """Tick params used in `plt.tick_params` or `im.axes.tick_params` to
    plot images without labels, borders etc..
    """
    return dict(axis='both', which='both',
                bottom='off', top='off', left='off', right='off',
                labelbottom='off', labelleft='off', labelright='off')
