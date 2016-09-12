
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

# 模型的初始化和导入
restart = True      # 是否要放弃之前生成的模型
tmp_path = '/home/multiangle/tmp'
if not os.path.exists(tmp_path):
    os.makedirs(tmp_path)
file_name = 'model.pkl'
file_path = os.path.join(tmp_path,file_name)
if restart or not os.path.exists(file_path):
    input_dim   = 2     # 输入层的维数
    output_dim  = 4     # 输出层的维数
    hidden_size = 10    # 潜在层cell个数
    Whx = np.random.randn(hidden_size, input_dim)
    Wyh = np.random.randn(output_dim, hidden_size)
    bh  = np.zeros((hidden_size, 1))
    by  = np.zeros((output_dim, 1))
else:
    tf = open(file_path,'rb')
    model = pickle.load(tf)
    tf.close()
    input_dim   = model['input_dim']
    output_dim  = model['output_dim']
    hidden_size = model['hidden_size']
    Whx         = model['Whx']
    Wyh         = model['Wyh']
    bh          = model['bh']
    by          = model['by']

def save():
    model = dict(
        input_dim   = input_dim,
        output_dim  = output_dim,
        hidden_size = hidden_size ,
        Whx         = Whx,
        Wyh         = Wyh,
        bh          = bh,
        by          = by
    )
    tf = open(file_path,'wb')
    pickle.dump(model,tf)
    tf.close()

def train(input, output, Whx, Wyh):
    """
    :param inputs:      list, 例如[1,2,3]
    :param outputs:     list, 例如[4,5,6]
    :return: void
    """
    # assert input.__len__()==input_dim
    # assert output.__len__()==output_dim

    # 生成输出值
    input = np.array([input]).T
    output = np.array([output]).T
    h = np.dot(Whx, input) + bh
    # h_z = np.dot(Whx, input) + bh  # z表示线性值
    # h_a = 1/(1+np.exp(-1*h_z))     # a表示经过激活函数以后的值, 这里是sigmoid
    y_a = np.dot(Wyh, h) + by

    # 生产梯度，差错传递
    c_y = y_a - output
    dWyh = np.dot(c_y, h.T)
    dby = c_y
    c_h = np.dot(Wyh.T, c_y)
    dWhx = np.dot(c_h, input.T)
    dbh = c_h
    # print(c_y)
    # print(c_h)

    return dWhx,dWyh,dby,dbh,c_y

# 输入是2个值 x,x^2
# 输出是4个值 3x^2+2x-5,10x-10, 2x^2-5x, 5
x = np.arange(0,30,0.2)
x2 = x*x
y_1 = 3*x2+2*x-5
y_2 = 10*x-10
y_3 = 2*x2-5*x
y_4 = np.ones((x.__len__()))

yout = None
mWhx, mWyh = np.zeros_like(Whx), np.zeros_like(Wyh)
mbh, mby = np.zeros_like(bh), np.zeros_like(by)
out = None
expect_out = None
d_abs = []
for i in range(x.__len__()):
    input = [x[i],x2[i]]
    output = [y_1[i],y_2[i],y_3[i],y_4[i]]
    dWhx,dWyh,dby,dbh,c_y = train(input,output,Whx,Wyh)

    learning_rate = 0.1
    d = []
    for param,dparam,mem in zip([Whx,Wyh,by,bh],
                                [dWhx,dWyh,dby,dbh],
                                [mWhx,mWyh,mby,mbh]):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8)
        # param += -learning_rate * dparam
    input_v = np.array([input]).T
    output_v = np.array([output]).T
    thisout = np.dot(Wyh,np.dot(Whx,input_v)+bh)+by
    if out!=None:
        out = np.append(out,thisout,axis=1)
    else:
        out = thisout
    if expect_out!=None:
        expect_out = np.append(expect_out,output_v,axis=1)
    else:
        expect_out = output_v

c_l = ['b','r','g','y']
for i in range(4):
    plt.plot(expect_out[i,:],c_l[i])
    plt.plot(out[i,:],c_l[i]+'--')
plt.show()






