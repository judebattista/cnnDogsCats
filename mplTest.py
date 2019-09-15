import matplotlib
guis = [i for i in matplotlib.rcsetup.interactive_bk]
for gui in guis:
    print('testing {0}'.format(gui))
    try:
        from matplotlib import pyplot as plt
        print('    {0} is available'.format(gui))
        plt.plot([1.5, 2.0, 2.5])
        fig = plt.gcf()
        fig.suptitle(gui)
        plt.show()
        print('Using .... {0}'.format(matplotlib.get_backend()))
    except:
        print('    {0} not found.'.format(gui))
    
