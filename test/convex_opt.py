
import matplotlib.pyplot as plt


def cal_theta_grad(x,y,xi,yi,ri):
    grad_x = -1*ri*(x-xi)/(cal_sqrt(x,y,xi,yi)**3+0.00001)
    grad_y = -1*ri*(y-yi)/(cal_sqrt(x,y,xi,yi)**3+0.00001)
    return [grad_x, grad_y]

def cal_mul_theta_grad(x,y,xi,yi,ri,xj,yj,rj):
    pre_item = -1*ri*rj/((cal_sqrt(x,y,xi,yi)**2)*(cal_sqrt(x,y,xj,yj)**2)+0.00001)
    ratio_i_j = cal_sqrt(x,y,xi,yi)/(cal_sqrt(x,y,xj,yj)+0.00001)
    grad_x = pre_item*((x-xi)/(ratio_i_j+0.00001)+(x-xj)*ratio_i_j)
    grad_y = pre_item*((y-yi)/(ratio_i_j+0.00001)+(y-yj)*ratio_i_j)
    return [grad_x, grad_y]

def cal_sqrt(x,y,xi,yi):
    res = ((x-xi)**2 + (y-yi)**2)**0.5
    return res

def cal_theta(x,y,xi,yi,ri):
    return ri/(((x-xi)**2+(y-yi)**2)**0.5+0.00001)

def cal_grad(x,y,data):
    # data : [x,y,r]
    gradx = 0
    grady = 0
    size = data.__len__()
    for item in data:
        tx, ty = cal_theta_grad(x,y,item[0],item[1],item[2])
        tmpx = 2*cal_theta(x,y,item[0],item[1],item[2])*tx
        tmpy = 2*cal_theta(x,y,item[0],item[1],item[2])*ty
        gradx += tmpx
        grady += tmpy

    for i in range(size):
        for j in range(i+1,size):
            tmpx, tmpy = cal_mul_theta_grad(x,y,data[i][0],data[i][1],data[i][2],data[j][0],data[j][1],data[j][2])
            gradx -= tmpx
            grady -= tmpy
    return [gradx, grady]

trace = []
x=10
y=15
data = [
    [0,0,10],
    [60,0,10],
    [30,30,10]
]

times=0
trace.append([x,y])
while times<1000 :
    gradx, grady = cal_grad(x,y,data)
    print(gradx,grady)
    lamb = 20
    x -= lamb*gradx
    y -= lamb*grady
    trace.append([x,y])
    times += 1
x_trace = [i[0] for i in trace]
y_trace = [i[1] for i in trace]
plt.plot(x_trace,y_trace,'.')
plt.show()