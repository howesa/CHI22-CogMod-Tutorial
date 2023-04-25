import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import animation

def animate_trace(trace, get_anim = False, x_lim = [-100,100], y_lim = [100,-100]):
    plt.close()
    interval = 0.05

    frames = len(trace)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, aspect='equal')
    ax.set_xlim(x_lim[0], x_lim[1])
    ax.set_ylim(y_lim[0], y_lim[1])


    axl = plt.gca()
    axl.invert_yaxis()

    veh1 = plt.Circle((0, 0), 2, fc='orange')
    veh2 = plt.Circle((0, 0), 2, fc='red')

    time_text = ax.text(-30, 40, "Time:", fontsize=8)
    dist_text = ax.text(-30, 35, "Dist:", fontsize=8)
    coll_text = ax.text(-30, 30, "Coll:", fontsize=8)

    def init():
        time_text.set_text('')
        dist_text.set_text('')
        coll_text.set_text('')
        return veh1, veh2, time_text, dist_text, coll_text

    def animate(i):
        if i == 0:
            ax.add_patch(veh1)
            ax.add_patch(veh2)
        t = trace[i][0]
        t = i * interval
        x = int(trace[i][1][0])
        y = int(trace[i][1][1])
        veh1.center = (x, y)
        x = int(trace[i][2][0])
        y = int(trace[i][2][1])
        veh2.center = (x, y)

        if trace[i][4]:
            veh2.set_color('r')

        time_text.set_text('Time {:1.2f}'.format(round(t,2)))
        d = round(trace[i][3],2)
        dist_text.set_text('Dist {:1.2f}'.format(d))
        coll_text.set_text('Coll {:1.5s}'.format(str(trace[i][4])))
        return veh1, veh2, time_text, dist_text, coll_text

    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=frames,
        repeat=False,
        interval=1000 * interval,
        blit=True,
    )

    if get_anim:
        return anim
    else:
        plt.show()
