import numpy as np
from sklearn.externals import joblib
import matplotlib.colors
import matplotlib.pyplot as plt 
import net_clean
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.bmp','.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])
    
def sigmoid(x):
    return (1/(1+np.exp(-x)))

def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
#对y 进行操作
def diftanh(x):
    return (1-np.power(x,2))
#对y进行操作
def dif(y):
    return y*(1-y)

def softmax(x):
    c = np.max(x)
    return np.exp(x-c)/np.sum(np.exp(x-c))

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
def net_predict(x,Hw,Ow):
    xm = x.reshape(-1,1)
    x = xm.copy()
    x = np.append(x,1)#加上权重
    x = x.reshape(-1,1)
    Hi = np.dot(x.T,Hw.T) #1*3
    #隐层神经元输出
    Ho1 = tanh(Hi)
    Ho = Ho1.copy()
    Ho = np.append(Ho,1)
    Ho = Ho.reshape(1,-1)
    Oi= np.dot(Ho,Ow.T)
    #输出层的输出
    Oo = softmax(Oi)
    y = (np.argmax(Oo))
    return y
def netjcs_predict(x,Hw,Ow):
    xm = x.reshape(-1,1)
    x = xm.copy()
    x = np.append(x,1)#加上权重
    x = x.reshape(-1,1)
    Hi = np.dot(x.T,Hw.T) #1*3
    #隐层神经元输出
    Ho1 = sigmoid(Hi)
    Ho = Ho1.copy()
    Ho = np.append(Ho,1)
    Ho = Ho.reshape(1,-1)
    Oi= np.dot(Ho,Ow.T)
    #输出层的输出
    Oo = softmax(Oi)
    y = (np.argmax(Oo))
    return y

def net_pro_jiaocha_5_elu_predict(xm,Hw):
    eluv = np.vectorize(elu)
    difeluv = np.vectorize(difelu)
    x = xm.copy()
    x = np.append(x,1)#加上权重
    x = x.reshape(-1,1)
    # print(x)

    #第一层隐层神经元总输入
    Hi1 = np.dot(x.T,Hw[0].T) #1*3
    #第一隐层神经元输出
    Ho11 = eluv(Hi1)#实际输出反向传播用
    # Ho11 = sigmoid(Hi1)
    Ho1 = Ho11.copy()
    Ho1 = np.append(Ho1,1)
    Ho1 = Ho1.reshape(1,-1)#准备传到下一层的输出
    
    #第2层隐层神经元总输入
    Hi2 = np.dot(Ho1,Hw[1].T) #1*3
    #第2隐层神经元输出
    Ho21 = eluv(Hi2)#实际输出 
    # Ho21 = sigmoid(Hi2)
    Ho2 = Ho21.copy()
    Ho2 = np.append(Ho2,1)
    Ho2 = Ho2.reshape(1,-1)#准备传到下一层的输出
    
    #第3层隐层神经元总输入
    Hi3 = np.dot(Ho2,Hw[2].T) #1*3
    #第3隐层神经元输出
    Ho31 = eluv(Hi3)#实际输出 
    # Ho31 = sigmoid(Hi3)
    Ho3 = Ho31.copy()
    Ho3 = np.append(Ho3,1)
    Ho3 = Ho3.reshape(1,-1)#准备传到下一层的输出

    #输出层的输入
    Oi= np.dot(Ho3,Hw[3].T)
    #输出层的输出
    Oo = softmax(Oi)
    return (np.argmax(Oo))

def net_pro_jiaocha_4_elu_predict(xm,Hw):
    eluv = np.vectorize(elu)
    difeluv = np.vectorize(difelu)
    x = xm.copy()
    x = np.append(x,1)#加上权重
    x = x.reshape(-1,1)

    #第一层隐层神经元总输入
    Hi1 = np.dot(x.T,Hw[0].T) #1*3
    #第一隐层神经元输出
    Ho11 = eluv(Hi1)#实际输出反向传播用
    # Ho11 = sigmoid(Hi1)
    Ho1 = Ho11.copy()
    Ho1 = np.append(Ho1,1)
    Ho1 = Ho1.reshape(1,-1)#准备传到下一层的输出
    
    #第2层隐层神经元总输入
    Hi2 = np.dot(Ho1,Hw[1].T) #1*3
    #第2隐层神经元输出
    Ho21 = eluv(Hi2)#实际输出 
    # Ho21 = sigmoid(Hi2)
    Ho2 = Ho21.copy()
    Ho2 = np.append(Ho2,1)
    Ho2 = Ho2.reshape(1,-1)#准备传到下一层的输出
    
    #输出层的输入
    Oi= np.dot(Ho2,Hw[2].T)
    #输出层的输出
    Oo = softmax(Oi)
    return (np.argmax(Oo))


#交叉熵为loss函数3层隐层为sigmoid 输出层softmax 
def net_pro_jiaocha(xin,yout,hiddennum,outputnum,learn_rate = 0.5,diedai = 10000,alpha = 0.95):  
    print('====================================================')
    print('学习速率：%.1f,迭代次数：%d'%(learn_rate,diedai)) 
    print('开始训练........')
    error = np.array([])
    diedaix = np.arange(0,diedai,50)
    h = np.zeros((12,1),dtype=np.uint8)
    stop_flag = 0
    #输入神经元个数 
    inputnum = xin.shape[1]
    # 初始化隐藏层权重
    Hw = np.random.uniform(low=-0.3, high=0.3, size=(hiddennum,inputnum+1))
    #初始化输出层权重
    Ow = np.random.uniform(low=-0.3, high=0.3, size=(outputnum,hiddennum+1))
    xnum = xin.shape[0]
    piandao1_before = 0
    piandao2_before = 0
    for i in range(diedai):
        index = np.random.randint(0,xnum,size= (1))
        # print(index)
        xm = xin[index[0]].reshape(-1,1)
        x = xm.copy()
        x = np.append(x,1)#加上权重
        x = x.reshape(-1,1)
        # print(x)
        y = h.copy()
        y[yout[index[0]]] = 1

        # y = yout[index[0]]
        # print(y)
    #隐层神经元总输入
        Hi = np.dot(x.T,Hw.T) #1*3
        #隐层神经元输出
        Ho1 = sigmoid(Hi)
        Ho = Ho1.copy()
        Ho = np.append(Ho,1)
        Ho = Ho.reshape(1,-1)
        # print('Ho',Ho.shape)
        #输出层的输入
        Oi= np.dot(Ho,Ow.T)
        #输出层的输出
        Oo = softmax(Oi)
        # print(Oo.shape)
        # print(Oo)
        #反向传播第一层
        #计算残差zk-tk*f'
        delta1 = Oo-y.T
        # print('delta1',delta1.shape)
        piandao1 = np.dot(delta1.T,Ho)#输出层权重的偏导数 用虚拟的Ho

        #更新输出层权重
        Ow = Ow - learn_rate*piandao1+ alpha*piandao1_before
        piandao1_before = piandao1.copy()
        # print('ow',Ow.shape)
        #反向传播第二层
        dif2 = dif(Ho1)#
        delta2 = dif2 *(np.dot(delta1,Ow[:,:-1]))#残差得用实际神经元的残差
        # print('delta2',delta2.shape)
        piandao2 = np.dot(delta2.T,x.T)#向前求偏导数用虚拟的x
        #更新隐层权重
        Hw = Hw - learn_rate*piandao2+alpha*piandao2_before
        piandao2_before = piandao2.copy()
        y1 = y.T
        erro = np.sum(np.power(y1-Oo,2))
        if i%50==0:
            erro1 = np.sum(np.power(y1-Oo,2))
            error = np.append(error,erro1)
            # print(y)
            # print(Oo)
        # if erro <0.001:
        #     stop_flag +=1
        # else:
        #     stop_flag = 0
        # if stop_flag>10:
        #     print('第%d次迭代，误差小于0.001,连续%d次'%(i,stop_flag))
        # if stop_flag > 80:
        #     break
    
    plt.plot(diedaix,error,color= 'red')
    plt.title('Mes Loss')
    plt.xlabel('迭代次数')
    plt.ylabel('均方损失')
    plt.show()
    print('训练完成(迭代次数：%d)'% (i+1))
    print('====================================================')
    return Hw,Ow

#加上阈值b之后的,均方误差为损失函数 隐层为tanh 输出层softmax
def net_pro(xin,yout,hiddennum,outputnum,learn_rate = 0.5,diedai = 10000,alpha = 0.95):  
    print('====================================================')
    print('学习速率：%.1f,迭代次数：%d'%(learn_rate,diedai)) 
    print('开始训练........')
    error = np.array([])
    diedaix = np.arange(0,diedai,50)
    h = np.zeros((12,1),dtype=np.uint8)
    stop_flag = 0
    #输入神经元个数 
    inputnum = xin.shape[1]
    # 初始化隐藏层权重
    Hw = np.random.uniform(low=-0.3, high=0.3, size=(hiddennum,inputnum+1))
    #初始化输出层权重
    Ow = np.random.uniform(low=-0.3, high=0.3, size=(outputnum,hiddennum+1))
    xnum = xin.shape[0]
    piandao1_before = 0
    piandao2_before = 0
    for i in range(diedai):

        index = np.random.randint(0,xnum,size= (1))
        # print(index)
        xm = xin[index[0]].reshape(-1,1)
        x = xm.copy()
        x = np.append(x,1)#加上权重
        x = x.reshape(-1,1)
        # print(x)
        y = h.copy()
        y[yout[index[0]]] = 1

        # y = yout[index[0]]
        # print(y)
    #隐层神经元总输入
        Hi = np.dot(x.T,Hw.T) #1*3
        #隐层神经元输出
        Ho1 = tanh(Hi)
        Ho = Ho1.copy()
        Ho = np.append(Ho,1)
        Ho = Ho.reshape(1,-1)
        # print('Ho',Ho.shape)
        #输出层的输入
        Oi= np.dot(Ho,Ow.T)
        #输出层的输出
        Oo = softmax(Oi)
        # print(Oo.shape)
        # print(Oo)
        #反向传播第一层
        #计算残差zk-tk*f'
        dif1 = dif(Oo)
        delta1 = (Oo-y.T)*dif1
        # print('delta1',delta1.shape)
        piandao1 = np.dot(delta1.T,Ho)#输出层权重的偏导数 用虚拟的Ho

        #更新输出层权重
        Ow = Ow - learn_rate*piandao1+ alpha*piandao1_before
        piandao1_before = piandao1.copy()
        # print('ow',Ow.shape)
        #反向传播第二层
        dif2 = diftanh(Ho1)#
        delta2 = dif2 *(np.dot(delta1,Ow[:,:-1]))#残差得用实际神经元的残差
        # print('delta2',delta2.shape)
        piandao2 = np.dot(delta2.T,x.T)#向前求偏导数用虚拟的x
        #更新隐层权重
        Hw = Hw - learn_rate*piandao2+alpha*piandao2_before
        piandao2_before = piandao2.copy()
        y1 = y.T
        erro = np.sum(np.power(y1-Oo,2))
            
 
            # print(y)
            # print(Oo)
        # if erro <0.001:
        #     stop_flag +=1
        # else:
        #     stop_flag = 0
        # if stop_flag>10:
        #     print('第%d次迭代，误差小于0.001,连续%d次'%(i,stop_flag))
        # if stop_flag > 80:
        #     break
        if i%50==0:
            erro1 = np.sum(np.power(y1-Oo,2))
            error = np.append(error,erro1)


            # print ('error:%f'% erro)

    plt.plot(diedaix,error,color= 'red')
    plt.title('Mes Loss')
    plt.xlabel('迭代次数')
    plt.ylabel('均方损失')
    plt.show()
    print('训练完成(迭代次数：%d)'% (i+1))
    print('====================================================')
    return Hw,Ow

#五层
def net_pro_jiaocha_5_elu(xin,yout,hiddennum0,hiddennum1,hiddennum2,outputnum,learn_rate = 0.5,diedai = 10000,alpha = 0):  
    eluv = np.vectorize(elu)
    difeluv = np.vectorize(difelu)

    error = np.array([])
    diedaix = np.arange(0,diedai,50)
        
    plt.figure(figsize=(8, 8), facecolor='w')
    print('====================================================')
    print('学习速率：%.1f,迭代次数：%d'%(learn_rate,diedai)) 
    print('开始训练........')
    h = np.zeros((outputnum,1),dtype=np.uint8)
    #输入神经元个数 
    inputnum = xin.shape[1]
    # 初始化3个隐藏层权重
    Hw1 = np.random.uniform(low=-0.3, high=0.3, size=(hiddennum0,inputnum+1))
    Hw2 = np.random.uniform(low=-0.3, high=0.3, size=(hiddennum1,hiddennum0+1))
    Hw3 = np.random.uniform(low=-0.3, high=0.3, size=(hiddennum2,hiddennum1+1))
    #初始化输出层权重
    Ow = np.random.uniform(low=-0.3, high=0.3, size=(outputnum,hiddennum2+1))

    xnum = xin.shape[0]
    piandao1_before = 0
    piandao2_before = 0
    piandao3_before = 0
    piandao4_before = 0
    for i in range(diedai):
        index = np.random.randint(0,xnum,size= (1))
        # print(index)
        xm = xin[index[0]].reshape(-1,1)
        x = xm.copy()
        x = np.append(x,1)#加上权重
        x = x.reshape(-1,1)
        # print(x)
        y = h.copy()
        y[yout[index[0]]] = 1

        #第一层隐层神经元总输入
        Hi1 = np.dot(x.T,Hw1.T) #1*3
        #第一隐层神经元输出
        Ho11 = eluv(Hi1)#实际输出反向传播用
        # Ho11 = sigmoid(Hi1)
        Ho1 = Ho11.copy()
        Ho1 = np.append(Ho1,1)
        Ho1 = Ho1.reshape(1,-1)#准备传到下一层的输出
        
        #第2层隐层神经元总输入
        Hi2 = np.dot(Ho1,Hw2.T) #1*3
        #第2隐层神经元输出
        Ho21 = eluv(Hi2)#实际输出 
        # Ho21 = sigmoid(Hi2)
        Ho2 = Ho21.copy()
        Ho2 = np.append(Ho2,1)
        Ho2 = Ho2.reshape(1,-1)#准备传到下一层的输出
        
        #第3层隐层神经元总输入
        Hi3 = np.dot(Ho2,Hw3.T) #1*3
        #第3隐层神经元输出
        Ho31 = eluv(Hi3)#实际输出 
        # Ho31 = sigmoid(Hi3)
        Ho3 = Ho31.copy()
        Ho3 = np.append(Ho3,1)
        Ho3 = Ho3.reshape(1,-1)#准备传到下一层的输出
    
        #输出层的输入
        Oi= np.dot(Ho3,Ow.T)
        #输出层的输出
        Oo = softmax(Oi)

        #反向传播第一层输出层
        #计算残差zk-tk*f'
        delta1 = Oo-y.T
        # print('delta1',delta1.shape)
        piandao1 = np.dot(delta1.T,Ho3)#输出层权重的偏导数 用虚拟的Ho

        #更新输出层权重
        Ow = Ow - learn_rate*piandao1+ alpha*piandao1_before
        piandao1_before = piandao1.copy()

        #更新第三隐层的权重 反向传播第二层
        dif2 = difeluv(Hi3)#
        # dif2 = dif(Ho31)
        delta2 = dif2 *(np.dot(delta1,Ow[:,:-1]))#残差得用实际神经元的残差
        # print('delta2',delta2.shape)
        piandao2 = np.dot(delta2.T,Ho2)#向前求偏导数用虚拟的x
        #更新3隐层权重
        Hw3 = Hw3 - learn_rate*piandao2+alpha*piandao2_before
        piandao2_before2 = piandao2.copy()

        #更新第2隐层的权重 反向传播第3层
        dif3 = difeluv(Hi2)#
        # dif3 = dif(Ho21)
        delta3 = dif3 *(np.dot(delta2,Hw3[:,:-1]))#残差得用实际神经元的残差
        # print('delta2',delta2.shape)
        piandao3 = np.dot(delta3.T,Ho1)#向前求偏导数用虚拟的x
        #更新2隐层权重
        Hw2 = Hw2 - learn_rate*piandao3+alpha*piandao3_before
        piandao3_before = piandao3.copy()

        #更新第1隐层的权重 反向传播第4层
        dif4 = difeluv(Hi1)#
        # dif4 = dif(Ho11) 
        delta4 = dif4 *(np.dot(delta3,Hw2[:,:-1]))#残差得用实际神经元的残差
        # print('delta2',delta2.shape)
        piandao4 = np.dot(delta4.T,x.T)#向前求偏导数用虚拟的x
        #更新1隐层权重
        Hw1 = Hw1 - learn_rate*piandao4+alpha*piandao4_before
        piandao4_before = piandao4.copy()


        y1 = y.T
        erro = np.sum(np.power(y1-Oo,2))

        if i%50==0:
            erro1 = np.sum(np.power(y1-Oo,2))
            error = np.append(error,erro1)


            # print ('error:%f'% erro)

    plt.plot(diedaix,error,color= 'red')
    plt.title('Mes Loss')
    plt.xlabel('迭代次数')
    plt.ylabel('均方损失')
    plt.show()

    print('训练完成(迭代次数：%d)'% (i+1))
    print('====================================================')
    return Hw1,Hw2,Hw3,Ow

#4层
def net_pro_jiaocha_4_sigmoid(xin,yout,hiddennum0,hiddennum1,outputnum,learn_rate = 0.5,diedai = 10000,alpha = 0):  
    error = np.array([])
    diedaix = np.arange(0,diedai,50)
    print('====================================================')
    print('学习速率：%.1f,迭代次数：%d'%(learn_rate,diedai)) 
    print('开始训练........')
    h = np.zeros((outputnum,1),dtype=np.uint8)
    #输入神经元个数 
    inputnum = xin.shape[1]
    # 初始化3个隐藏层权重
    Hw1 = np.random.uniform(low=-0.3, high=0.3, size=(hiddennum0,inputnum+1))
    Hw2 = np.random.uniform(low=-0.3, high=0.3, size=(hiddennum1,hiddennum0+1))
    #初始化输出层权重
    Ow = np.random.uniform(low=-0.3, high=0.3, size=(outputnum,hiddennum1+1))

    xnum = xin.shape[0]
    piandao1_before = 0
    piandao2_before = 0
    piandao3_before = 0
    for i in range(diedai):
        index = np.random.randint(0,xnum,size= (1))
        # print(index)
        xm = xin[index[0]].reshape(-1,1)
        x = xm.copy()
        x = np.append(x,1)#加上权重
        x = x.reshape(-1,1)
        # print(x)
        y = h.copy()
        y[yout[index[0]]] = 1

        #第一层隐层神经元总输入
        Hi1 = np.dot(x.T,Hw1.T) #1*3
        #第一隐层神经元输出
        Ho11 = sigmoid(Hi1)#实际输出反向传播用
        # Ho11 = sigmoid(Hi1)
        Ho1 = Ho11.copy()
        Ho1 = np.append(Ho1,1)
        Ho1 = Ho1.reshape(1,-1)#准备传到下一层的输出
        
        #第2层隐层神经元总输入
        Hi2 = np.dot(Ho1,Hw2.T) #1*3
        #第2隐层神经元输出
        Ho21 = sigmoid(Hi2)#实际输出 
        # Ho21 = sigmoid(Hi2)
        Ho2 = Ho21.copy()
        Ho2 = np.append(Ho2,1)
        Ho2 = Ho2.reshape(1,-1)#准备传到下一层的输出
        
    
        #输出层的输入
        Oi= np.dot(Ho2,Ow.T)
        #输出层的输出
        Oo = softmax(Oi)

        #反向传播第一层输出层
        #计算残差zk-tk*f'
        delta1 = Oo-y.T
        # print('delta1',delta1.shape)
        piandao1 = np.dot(delta1.T,Ho2)#输出层权重的偏导数 用虚拟的Ho

        #更新输出层权重
        Ow = Ow - learn_rate*piandao1+ alpha*piandao1_before
        piandao1_before = piandao1.copy()


        #更新第2隐层的权重 反向传播第2层
        dif3 = dif(Ho21)#
        # dif3 = dif(Ho21)
        delta3 = dif3 *(np.dot(delta1,Ow[:,:-1]))#残差得用实际神经元的残差
        # print('delta2',delta2.shape)
        piandao2 = np.dot(delta3.T,Ho1)#向前求偏导数用虚拟的x
        #更新2隐层权重
        Hw2 = Hw2 - learn_rate*piandao2+alpha*piandao2_before
        piandao2_before = piandao2.copy()

        #更新第1隐层的权重 反向传播第3层
        dif4 = dif(Ho11)#
        # dif4 = dif(Ho11) 
        delta4 = dif4 *(np.dot(delta3,Hw2[:,:-1]))#残差得用实际神经元的残差
        # print('delta2',delta2.shape)
        piandao3 = np.dot(delta4.T,x.T)#向前求偏导数用虚拟的x
        #更新1隐层权重
        Hw1 = Hw1 - learn_rate*piandao3
        piandao3_before = piandao3.copy()


        y1 = y.T
        erro = np.sum(np.power(y1-Oo,2))

        if i%50==0:
            erro1 = np.sum(np.power(y1-Oo,2))
            error = np.append(error,erro1)


            # print ('error:%f'% erro)

    plt.plot(diedaix,error,color= 'red')
    plt.title('Mes Loss')
    plt.xlabel('迭代次数')
    plt.ylabel('均方损失')
    plt.show()

    print('训练完成(迭代次数：%d)'% (i+1))
    print('====================================================')
    return Hw1,Hw2,Ow

#4层
def net_pro_jiaocha_4_elu(xin,yout,hiddennum0,hiddennum1,outputnum,learn_rate = 0.5,diedai = 10000,alpha = 0):  
    eluv = np.vectorize(elu)
    difeluv = np.vectorize(difelu)

    error = np.array([])
    diedaix = np.arange(0,diedai,50)
        
    plt.figure(figsize=(8, 8), facecolor='w')

    print('====================================================')
    print('学习速率：%.1f,迭代次数：%d'%(learn_rate,diedai)) 
    print('开始训练........')
    h = np.zeros((outputnum,1),dtype=np.uint8)
    #输入神经元个数 
    inputnum = xin.shape[1]
    # 初始化3个隐藏层权重
    Hw1 = np.random.uniform(low=-0.3, high=0.3, size=(hiddennum0,inputnum+1))
    Hw2 = np.random.uniform(low=-0.3, high=0.3, size=(hiddennum1,hiddennum0+1))
    #初始化输出层权重
    Ow = np.random.uniform(low=-0.3, high=0.3, size=(outputnum,hiddennum1+1))

    xnum = xin.shape[0]
    piandao1_before = 0
    piandao2_before = 0
    piandao3_before = 0
    for i in range(diedai):
        index = np.random.randint(0,xnum,size= (1))
        # print(index)
        xm = xin[index[0]].reshape(-1,1)
        x = xm.copy()
        x = np.append(x,1)#加上权重
        x = x.reshape(-1,1)
        # print(x)
        y = h.copy()
        y[yout[index[0]]] = 1

        #第一层隐层神经元总输入
        Hi1 = np.dot(x.T,Hw1.T) #1*3
        #第一隐层神经元输出
        Ho11 = eluv(Hi1)#实际输出反向传播用
        # Ho11 = sigmoid(Hi1)
        Ho1 = Ho11.copy()
        Ho1 = np.append(Ho1,1)
        Ho1 = Ho1.reshape(1,-1)#准备传到下一层的输出
        
        #第2层隐层神经元总输入
        Hi2 = np.dot(Ho1,Hw2.T) #1*3
        #第2隐层神经元输出
        Ho21 = eluv(Hi2)#实际输出 
        # Ho21 = sigmoid(Hi2)
        Ho2 = Ho21.copy()
        Ho2 = np.append(Ho2,1)
        Ho2 = Ho2.reshape(1,-1)#准备传到下一层的输出
        
    
        #输出层的输入
        Oi= np.dot(Ho2,Ow.T)
        #输出层的输出
        Oo = softmax(Oi)

        #反向传播第一层输出层
        #计算残差zk-tk*f'
        delta1 = Oo-y.T
        # print('delta1',delta1.shape)
        piandao1 = np.dot(delta1.T,Ho2)#输出层权重的偏导数 用虚拟的Ho

        #更新输出层权重
        Ow = Ow - learn_rate*piandao1+ alpha*piandao1_before
        piandao1_before = piandao1.copy()


        #更新第2隐层的权重 反向传播第2层
        dif3 = difeluv(Hi2)#
        # dif3 = dif(Ho21)
        delta3 = dif3 *(np.dot(delta1,Ow[:,:-1]))#残差得用实际神经元的残差
        # print('delta2',delta2.shape)
        piandao2 = np.dot(delta3.T,Ho1)#向前求偏导数用虚拟的x
        #更新2隐层权重
        Hw2 = Hw2 - learn_rate*piandao2+alpha*piandao2_before
        piandao2_before = piandao2.copy()

        #更新第1隐层的权重 反向传播第3层
        dif4 = difeluv(Hi1)#
        # dif4 = dif(Ho11) 
        delta4 = dif4 *(np.dot(delta3,Hw2[:,:-1]))#残差得用实际神经元的残差
        # print('delta2',delta2.shape)
        piandao3 = np.dot(delta4.T,x.T)#向前求偏导数用虚拟的x
        #更新1隐层权重
        Hw1 = Hw1 - learn_rate*piandao3
        piandao3_before = piandao3.copy()


        y1 = y.T
        
        if i%50==0:
            erro1 = np.sum(np.power(y1-Oo,2))
            error = np.append(error,erro1)


            # print ('error:%f'% erro)

    plt.plot(diedaix,error,color= 'red')
    plt.title('Mes Loss')
    plt.xlabel('迭代次数')
    plt.ylabel('均方损失')
    plt.show()
    print('训练完成(迭代次数：%d)'% (i+1))
    print('====================================================')
    return Hw1,Hw2,Ow

#3层
def net_pro_jiaocha_3_elu(xin,yout,hiddennum,outputnum,learn_rate = 0.5,diedai = 10000,alpha = 0.95):  
    eluv = np.vectorize(elu)
    difeluv = np.vectorize(difelu)
    error = np.array([])
    diedaix = np.arange(0,diedai,50)
    print('====================================================')
    print('学习速率：%.1f,迭代次数：%d'%(learn_rate,diedai)) 
    print('开始训练........')
    h = np.zeros((12,1),dtype=np.uint8)
    stop_flag = 0
    #输入神经元个数 
    inputnum = xin.shape[1]
    # 初始化隐藏层权重
    Hw = np.random.uniform(low=-0.3, high=0.3, size=(hiddennum,inputnum+1))
    #初始化输出层权重
    Ow = np.random.uniform(low=-0.3, high=0.3, size=(outputnum,hiddennum+1))
    xnum = xin.shape[0]
    piandao1_before = 0
    piandao2_before = 0
    for i in range(diedai):
        index = np.random.randint(0,xnum,size= (1))
        # print(index)
        xm = xin[index[0]].reshape(-1,1)
        x = xm.copy()
        x = np.append(x,1)#加上权重
        x = x.reshape(-1,1)
        # print(x)
        y = h.copy()
        y[yout[index[0]]] = 1

        # y = yout[index[0]]
        # print(y)
    #隐层神经元总输入
        Hi = np.dot(x.T,Hw.T) #1*3
        #隐层神经元输出
        Ho1 = eluv(Hi)
        Ho = Ho1.copy()
        Ho = np.append(Ho,1)
        Ho = Ho.reshape(1,-1)
        # print('Ho',Ho.shape)
        #输出层的输入
        Oi= np.dot(Ho,Ow.T)
        #输出层的输出
        Oo = softmax(Oi)
        # print(Oo.shape)
        # print(Oo)
        #反向传播第一层
        #计算残差zk-tk*f'
        delta1 = Oo-y.T
        # print('delta1',delta1.shape)
        piandao1 = np.dot(delta1.T,Ho)#输出层权重的偏导数 用虚拟的Ho

        #更新输出层权重
        Ow = Ow - learn_rate*piandao1+ alpha*piandao1_before
        piandao1_before = piandao1.copy()
        # print('ow',Ow.shape)
        #反向传播第二层
        dif2 = difeluv(Hi)#
        delta2 = dif2 *(np.dot(delta1,Ow[:,:-1]))#残差得用实际神经元的残差
        # print('delta2',delta2.shape)
        piandao2 = np.dot(delta2.T,x.T)#向前求偏导数用虚拟的x
        #更新隐层权重
        Hw = Hw - learn_rate*piandao2+alpha*piandao2_before
        piandao2_before = piandao2.copy()
        y1 = y.T
        erro = np.sum(np.power(y1-Oo,2))
        if i%50==0:
            erro1 = np.sum(np.power(y1-Oo,2))
            error = np.append(error,erro1)


            # print ('error:%f'% erro)

    plt.plot(diedaix,error,color= 'red')
    plt.title('Mes Loss')
    plt.xlabel('迭代次数')
    plt.ylabel('均方损失')
    plt.show()
            # print(y)
            # print(Oo)
        # if erro <0.001:
        #     stop_flag +=1
        # else:
        #     stop_flag = 0
        # if stop_flag>10:
        #     print('第%d次迭代，误差小于0.001,连续%d次'%(i,stop_flag))
        # if stop_flag > 80:
        #     break
    print('训练完成(迭代次数：%d)'% (i+1))
    print('====================================================')
    return Hw,Ow


if __name__ == "__main__":
    matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    # #训练
    x = np.load('tarin_data.npy')
    # print(x.shape)
    y = np.load('train_label.npy')
    # print(y.shape)
    learn_rate = 0.002
    # diedai = 70000
    # x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.7)
    # Hw = net_pro_jiaocha_5_elu(x_train,y_train,8,8,8,12,learn_rate= learn_rate,diedai=diedai,alpha=0)#0.002,70000,0

    # y_test_hat = net_clean.net_pro_jiaocha_predict(x_test,Hw,func = 'elu',loss = 'mes')
    # print('train_set准确率：', accuracy_score(y_train, net_pro_jiaocha_predict(x_train,Hw,func = 'elu',loss = 'jcs')))
    # print('valid_set准确率：', accuracy_score(y_test,y_test_hat))


    # Hw = net_pro_jiaocha_4_elu(x,y,10,10,12,learn_rate= learn_rate,diedai=diedai,alpha=0)#0.01,20000,0

    # Hw,Ow = net_pro(x,y,20,12,learn_rate=learn_rate ,diedai=diedai,alpha=0)#0.2,20000,0
    # Hw,Ow = net_pro_jiaocha(x,y,20,12,learn_rate=learn_rate ,diedai=diedai,alpha=0)#0.2,20000,0
    # Hw1,d,Ow = net_pro_jiaocha_4_sigmoid(x,y,10,10,12,learn_rate=learn_rate ,diedai=diedai,alpha=0)#0.10,40000,0

    # Hw,Ow = net_pro_jiaocha_3_elu(x,y,20,12,learn_rate= learn_rate,diedai=diedai,alpha=0)#0.01,40000,0

    # joblib.dump(Hw,'Hw'+'_'+str(learn_rate)+'_'+str(diedai))
    # joblib.dump(Ow,'Ow'+'_'+str(learn_rate)+'_'+str(diedai))
    # x = np.array([-1,-2,2,3])
    # x = x.reshape(1,-1)
    # func = np.vectorize(dif3)
    # print(func(x).shape)
    # print(y.shape)

# def net(xin,yout,hiddennum,outputnum,learn_rate = 0.5,diedai = 10000):  
#     print('====================================================')
#     print('学习速率：%.1f,迭代次数：%d'%(learn_rate,diedai)) 
#     print('开始训练........')
#     h = np.zeros((12,1),dtype=np.uint8)
#     stop_flag = 0
#     #输入神经元个数 
#     inputnum = xin.shape[1]
#     # 初始化隐藏层权重3个隐藏层神经元
#     Hw = np.random.rand(hiddennum,inputnum)
#     #初始化输出层权重1个输出层神经元
#     Ow = np.random.rand(outputnum,hiddennum)
#     xnum = xin.shape[0]
#     for i in range(diedai):
#         index = np.random.randint(0,xnum,size= (1))
#         # print(index)
#         x = xin[index[0]].reshape(-1,1)
#         y = h.copy()
#         y[yout[index[0]]] = 1

#         # y = yout[index[0]]
#         # print(y)
#     #隐层神经元总输入
#         Hi = np.dot(x.T,Hw.T) #1*3
#         #隐层神经元输出
#         Ho = tanh(Hi)
#         # print('Ho',Ho.shape)
#         #输出层的输入
#         Oi= np.dot(Ho,Ow.T)
#         #输出层的输出
#         Oo = softmax(Oi)
#         # print(Oo.shape)
#         # print(Oo)
#         #反向传播第一层
#         #计算残差zk-tk*f'
#         dif1 = dif(Oo)
#         delta1 = (Oo-y.T)*dif1
#         # print('delta1',delta1.shape)
#         piandao1 = np.dot(delta1.T,Ho)#输出层权重的偏导数
#         #更新输出层权重
#         Ow = Ow - learn_rate*piandao1
#         # print('ow',Ow.shape)
#         #反向传播第二层
#         dif2 = diftanh(Ho)
#         delta2 = dif2 *(np.dot(delta1,Ow))#残差
#         # print('delta2',delta2.shape)
#         piandao2 = np.dot(delta2.T,x.T)
#         #更新隐层权重
#         Hw = Hw - learn_rate*piandao2
#         y1 = y.T
#         erro = np.sum(np.power(y1-Oo,2))
#         if i%500==0:
#             print ('error:%f'% erro)
#             # print(y)
#             # print(Oo)
#         # if erro <0.001:
#         #     stop_flag +=1
#         # else:
#         #     stop_flag = 0
#         # if stop_flag>10:
#         #     print('第%d次迭代，误差小于0.001,连续%d次'%(i,stop_flag))
#         # if stop_flag > 80:
#         #     break
#     print('训练完成(迭代次数：%d)'% (i+1))
#     print('====================================================')
#     return Hw,Ow








