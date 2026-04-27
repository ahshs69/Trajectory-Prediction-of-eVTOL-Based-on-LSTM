# -*- coding: gbk -*-
'''图表绘制'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
from mpl_toolkits.mplot3d import Axes3D


def animation(dataset,num_poses=252):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    line,=ax.plot([],[],[],markersize=4,color='blue',label='Trajectory')
    ax.set_title("3D_Animation")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    def init():
        line.set_data([],[])
        line.set_3d_properties([])
        return line
    def update(frame):
        line.set_data(dataset["tx"][:frame],dataset["ty"][:frame])
        line.set_3d_properties(dataset["tz"][:frame])
        ax.set_xlim(min(dataset["tx"])-1,max(dataset["tx"])+1)
        ax.set_ylim(min(dataset["ty"])-1,max(dataset["ty"])+1)
        ax.set_zlim(min(dataset["tz"])-1,max(dataset["tz"])+1)
        return line
    ani=FuncAnimation(fig,update,frames=num_poses,init_func=init,blit=False)
    ani.save(r'C:\Users\34362\Desktop\LSTM\results\animation_gt.gif',writer=PillowWriter(fps=30))
    plt.show()
    
    
    
def animation_double(idx,dataset1,dataset2):
    real_data = np.array(dataset1)
    pred_data = dataset2
    

    total_frames = len(real_data)
    print("总帧数:", total_frames)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    line_real, = ax.plot([],[],[],markersize=4,color='blue',label='real')
    line_pred, = ax.plot([], [], [], 'r-', linewidth=2, label='Pred')

    ax.set_title("3D_Animation")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.legend()

    def init():
        line_real.set_data([],[])
        line_real.set_3d_properties([])
        line_pred.set_data([],[])
        line_pred.set_3d_properties([])
        return line_real,line_pred

    def update(frame):
        x_r = real_data[:frame, 1]
        y_r = real_data[:frame, 2]
        z_r = real_data[:frame, 3]
        line_real.set_data(x_r, y_r)
        line_real.set_3d_properties(z_r)


        ax.set_xlim( real_data[:,1].min()-5, real_data[:,1].max()+5 )
        ax.set_ylim( real_data[:,2].min()-5, real_data[:,2].max()+5 )
        ax.set_zlim( real_data[:,3].min()-5, real_data[:,3].max()+5 )
                        
        pred = pred_data[frame]
        line_pred.set_data(pred[:,0], pred[:,1])
        line_pred.set_3d_properties(pred[:,2])

        return line_real,line_pred

    ani = FuncAnimation(
        fig,
        update,
        frames=total_frames,
        init_func=init,  
        blit=False,     
    
    )

    ani.save(
    fr'C:\Users\34362\Desktop\LSTM\results\animation_gt_{idx}.gif',
    writer=PillowWriter(fps=15))

    plt.show()


def plot_loss_curve(train_losses, val_losses, save_path="loss_curve.png"):


    plt.figure(figsize=(10, 6))
    
    # 画曲线
    plt.plot(train_losses, label="Train Loss", color="#3498db", linewidth=2)
    plt.plot(val_losses, label="Val Loss", color="#e74c3c", linewidth=2)
    
    # 样式
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Train & Validation Loss Curve", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f" Loss 折线图已保存：{save_path}")