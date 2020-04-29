import numpy as np
from sklearn.externals import joblib
import matplotlib.colors
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib.pyplot import MultipleLocator
#以交叉熵作为损失函数，激活函数可选为elu sigmoid 层数任意选择 注意学习速率的问题
#也可以用最小均方误差作为损失函数，激活函数选用的是tanh

def difmessoftmax(z,y):
    z = z.reshape(1,-1)
    y = y.reshape(1,-1)
    d1 = (z-y)*z*(1-z)
    d2 = np.array([])
    for i in range(len(z[0])):
        z1 = z.copy()
        y1 = y.copy()
        zi = z1[0][i]
        z1 = np.delete(z1,i)
        y1 = np.delete(y1,i)
        h = np.sum((z1-y1)*z1*zi)
        d2 = np.append(d2,h)
    d2 = d2.reshape(1,-1)
    delta = d1-d2
    return delta
def softmax(x):
    c = np.max(x)
    return np.exp(x-c)/np.sum(np.exp(x-c))
def difsoftmax(y):
    return y*(1-y)
#alpha = 0.5
def elu(x):
    if x<0:
        return 0.5*(np.exp(x)-1)
        # return 0
    else:
        return x
#对输入的x进行操作
def difelu(x):
    if x>=0:
        return 1.0
    else:
        return 0.5*np.exp(x)
        # return 0

def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
def sigmoid(x):
    return (1/(1+np.exp(-x)))
#对y 进行操作
def diftanh(x):
    return (1-np.power(x,2))

def difsigmoid(y):
    return y*(1-y)

def net_pro_jiaocha_predict(x,Hw,func = 'elu',loss = 'jcs'):
    leng = len(Hw)
    if loss == 'mes':
        func = 'tanh'
    if func == 'elu':
        eluv = np.vectorize(elu)
    elif func == 'sigmoid':
        eluv = np.vectorize(sigmoid)
    elif func == 'tanh':
        eluv = np.vectorize(tanh)
    y = np.array([],dtype = np.uint8)
    for xin in x:
        xm = xin.reshape(-1,1)
        x = xm.copy()
        x = np.append(x,1)#加上权重
        x = x.reshape(-1,1)
        #正向传播
        for i in range(leng):
            if i >= 0 and i<leng-1:#前几层
                if i == 0:
                    Him = np.dot(x.T,Hw[0].T)
                else:
                    Him = np.dot(Honext,Hw[i].T)
                Hom = eluv(Him)
                Honext = Hom.copy()
                Honext = np.append(Honext,1)
                Honext = Honext.reshape(1,-1)#准备传到下一层的输出
            elif i == leng-1:#最后一层
                Him = np.dot(Honext,Hw[i].T)
                Hom = softmax(Him)
                y = np.append(y,np.argmax(Hom))
    return y 


def net_pro_plus(xin,yout,x_test,y_test,*num,learn_rate = 0.5,diedai = 10000,alpha = 0,func = 'elu',loss = 'jcs'):  
    if loss == 'mes':
        func = 'tanh'

    if func == 'elu':
        eluv = np.vectorize(elu)
        difeluv = np.vectorize(difelu)
    elif func == 'sigmoid':
        eluv = np.vectorize(sigmoid)
        difeluv = np.vectorize(difsigmoid)
    elif func == 'tanh':
        eluv = np.vectorize(tanh)
        difeluv = np.vectorize(diftanh)

    error = np.array([])
    acc_train = np.array([])
    acc_valid = np.array([])
    diedaix = np.arange(0,diedai,50)
   
    fig = plt.figure(figsize=(12, 8), facecolor='w')
    if loss == 'mes':
        name = 'MSE'
    else:
        name = 'CEE'
    print('===========================================================')
    print('损失函数：%s,激活函数：%s,学习速率：%.3g,迭代次数：%d'%(name,func,learn_rate,diedai)) 
    print('开始训练........')
    leng = len(num[0])
    h = np.zeros((num[0][leng-1],1),dtype=np.uint8)
    #输入神经元个数 
    inputnum = xin.shape[1]

    #权重
    Hw = list([])
    # 初始化各层权重
    for i in range(leng):
        if i == 0:
            Hw.append(np.random.uniform(low=-0.3, high=0.3, size=(num[0][i],inputnum+1)))
        else:
            Hw.append(np.random.uniform(low=-0.3, high=0.3, size=(num[0][i],num[0][i-1]+1)))

    xnum = xin.shape[0]
    piandao_before = list(0 for _ in range(leng))
    for j in range(diedai):
        if j>0 and (j%1000==0):
            learn_rate = learn_rate/1.5
        #选取数据打上标签
        index = np.random.randint(0,xnum,size= (1))
        xm = xin[index[0]].reshape(-1,1)
        x = xm.copy()
        x = np.append(x,1)#加上权重
        x = x.reshape(-1,1)
        y = h.copy()
        y[yout[index[0]]] = 1
        
        #正向传播
        Hi = list([])#保存实际输入
        Ho = list([])#保存实际输出
        Ho_virture = list([])#虚拟输出
        Ho_virture.append(x.T)
        for i in range(leng):
            if i >= 0 and i<leng-1:#前几层
                if i == 0:
                    Him = np.dot(x.T,Hw[0].T)
                    Hi.append(Him)
                else:
                    Him = np.dot(Honext,Hw[i].T)
                    Hi.append(Him)
                Hom = eluv(Him)
                Ho.append(Hom)
                Honext = Hom.copy()
                Honext = np.append(Honext,1)
                Honext = Honext.reshape(1,-1)#准备传到下一层的输出
                Ho_virture.append(Honext)
            elif i == leng-1:#最后一层
                Him = np.dot(Honext,Hw[i].T)
                Hi.append(Him)
                Hom = softmax(Him)
                Ho.append(Hom)
        
        #反向传播
        for i in range(leng)[::-1]:
            if i == leng-1:
                if loss == 'jcs':
                    delta = Ho[i]-y.T
                elif loss == 'mes':
                    #速度快省略的做法
                    dif1 = difsoftmax(Ho[i])#最后一层是对sigmoid 求导
                    delta = (Ho[i]-y.T)*dif1
                    #速度慢一些正规做法
                    # delta = difmessoftmax(Ho[i],y)
            else:
                if loss == 'jcs':
                    if func == 'elu':
                        dif  = difeluv(Hi[i])
                    elif func == 'sigmoid':
                        dif = difeluv(Ho[i])
                    elif func == 'tanh':
                        dif = difeluv(Ho[i])
                elif loss == 'mes':
                    dif = difeluv(Ho[i])
                delta = dif * (np.dot(delta,Hw[i+1][:,:-1]))

            piandao = np.dot(delta.T,Ho_virture[i])
            Hw[i] = Hw[i] - learn_rate*piandao+ alpha*piandao_before[i]
            piandao_before[i] = piandao.copy()         
        y1 = y.T

        if j%50==0:
            #交叉验证
            y_test_hat = net_pro_jiaocha_predict(x_test,Hw,func = func,loss = loss)
            acc_t = accuracy_score(y_train, net_pro_jiaocha_predict(x_train,Hw,func = func,loss = loss))
            acc_v = accuracy_score(y_test,y_test_hat)
            if loss == 'mes':
                erro1 = np.sum(np.power(y1-Ho[leng-1],2))
            elif loss == 'jcs':
                erro1 = -np.sum(y1*np.log(Ho[leng-1]))
            error = np.append(error,erro1)
            acc_valid = np.append(acc_valid,acc_v)
            acc_train = np.append(acc_train,acc_t)
        
    ax1 = fig.add_subplot(111)



    ax2 = ax1.twinx()

    ax=plt.gca()
    y_major_locator=MultipleLocator(0.05)
    ax.yaxis.set_major_locator(y_major_locator)

    l1= ax1.plot(diedaix,error,color= 'red',label = 'training loss')
    if loss == 'jcs':
        ax1.set_ylabel('交叉熵损失')
    else:
        ax1.set_ylabel('均方损失')


    l2 = ax2.plot(diedaix,acc_train,color = 'green',label='accuracy of train_set')
    l3 = ax2.plot(diedaix,acc_valid,color = 'blue',label = 'accuracy of valid_set')

    ax2.set_ylabel('准确率')
    ax1.set_xlabel('迭代次数')



    ax1.grid(b=True, ls=':', color='k',axis = 'x')
    ax2.grid(b=True, ls=':', color='k')
    ax2.set_title(name+' Loss\n'+ func+'激活函数'+' '+str(leng+1)+'层\n'+'learn_rate:'+str(learn_rate))
    ins = l1+l2+l3
    labs = [l.get_label() for l in ins]
    ax2.legend(ins,labs,loc = 'center right')
    # ax2.legend(ins,labs,bbox_to_anchor = (1.05,1),loc=2,borderaxespad = 0)
    # fig.subplots_adjust(right = 0.8)
    print('训练完成(迭代次数：%d)'% (j+1))
    print('train_set准确率：',acc_t )
    print('valid_set准确率：',acc_v)
    print('===========================================================')

    if acc_t> 0.98 and acc_v > 0.98:
        acc_tstr = '{:.3f}'.format(acc_t)
        acc_vstr = '{:.3f}'.format(acc_v)
        pra = ''
        for x in num:
            pra = pra+'_'+str(x)
        name_file  = 'w'+'_'+loss+'_Pra'+pra+'_Acc'+acc_tstr+'_'+acc_vstr+'_'+func
        joblib.dump(Hw,name_file)
        print('保存完毕')
        print('===========================================================') 
    plt.show()

    return Hw


if __name__ == "__main__":
    matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    learn_rate = 0.3
    diedai = 4000# #训练
    loss = 'jcs'
    func = 'sigmoid'
    num = [20,12]
    
    x = np.load('tarin_data.npy')
    y = np.load('train_label.npy')
    xy = np.concatenate((x,y),axis= 1)
    np.random.shuffle(xy)
    xdata = xy[:,:-1]
    rydata = xy[:,-1]
    ydata = rydata.astype(np.uint8)
    ydata = ydata.reshape(-1,1)

    x_train, x_test, y_train, y_test = train_test_split(xdata, ydata, random_state=1, train_size=0.7)
    Hw = net_pro_plus(x_train,y_train,x_test,y_test,num,learn_rate= learn_rate,diedai=diedai,alpha=0,func=func,loss =loss)#0.002,70000,0
   
    # #交叉验证
    # y_test_hat = net_pro_jiaocha_predict(x_test,Hw,func = 'elu',loss = loss)
    # acc_t = accuracy_score(y_train, net_pro_jiaocha_predict(x_train,Hw,func = 'elu',loss = loss))
    # acc_v = accuracy_score(y_test,y_test_hat)
    # acc_train = '{:.3f}'.format(acc_t)
    # acc_valid = '{:.3f}'.format(acc_v)
    # print('train_set准确率：',acc_train )
    # print('valid_set准确率：',acc_valid)
    # print('===========================================================')

    # if acc_t> 0.98 and acc_v > 0.98:
    #     pra = ''
    #     for x in num:
    #         pra = pra+'_'+str(x)
    #     name_file  = 'w'+'_'+str(learn_rate)+'_'+str(diedai)+'_'+loss+'_Pra'+pra+'_Acc'+acc_train+'_'+acc_valid
    #     joblib.dump(Hw,name_file)
    #     print('保存完毕')
    #     print('===========================================================') 
    
    
    
    
    # a = np.array([1,2,3])
    # a = a.reshape(1,-1)
    # y = np.array([0,0,1])
    # y =y.reshape(1,-1)
    # h = difmessoftmax(a,y)
    # print(h)
    # # print(len(a[0]))