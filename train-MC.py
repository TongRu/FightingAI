'''
2020/6/01
Tong Ru
train —— traditional MC
'''

import random
import numpy as np
from fightingice_env import FightingiceEnv
import scipy.io as io

if __name__ == '__main__':
    env = FightingiceEnv(port=4242)
    # for windows user, port parameter is necessary because port_for library does not work in windows
    # for linux user, you can omit port parameter, just let env = FightingiceEnv()

    env_args = ["--fastmode", "--grey-bg", "--inverted-player", "1", "--mute"]#测试时采用此模式
    # this mode let two players have infinite hp, their hp in round can be negative
    # you can close the window display functional by using the following mode
    # env_args = ["--fastmode", "--disable-window", "--grey-bg", "--inverted-player", "1", "--mute"]#训练时采用此模式


    gamma=0.95
    # alpha=0.1
    alpha=0.01
    epsilon=0.1
    w=np.zeros((144*40,1))#初始权重
    p=0
    n=0

    N=500#训练次数
    NumWin=0
    RewardData=[]

    act = random.randint(0, 39)#初始动作

    A=np.zeros((40,144*40))
    for i in range(0,40):
        A[i,i*144:(i+1)*144]=1

    while True:
        obs = env.reset(env_args=env_args)
        reward, done, info = 0, False, None
        n=n+1#局数+1
        r=0#记录每局的reward
        perFeature=np.empty(shape=[0,144*40])
        perReward=np.empty(shape=[0,1])
        T=np.empty(shape=[0,1])
        m=0

        while not done:
            m=m+1
            # TODO: or you can design with your RL algorithm to choose action [act] according to game state [obs]
            X=np.tile(obs,(40,40))
            X=X*A
            X=X / np.array([X.max(axis=1)]).T
            act=np.argmax(np.dot(X,w))
            if random.random()<epsilon:
                act = random.randint(0, 39)
            else:
                pass
            x=np.zeros((1,144*40))
            x[0,144*act:144*(act+1)]=np.array([obs])
            new_obs, reward, done, info = env.step(act)

            if not done:
                # TODO: (main part) learn with data (obs, act, reward, new_obs)
                # X=np.tile(new_obs,(40,40))
                # X=X*A
                # X=X / np.array([X.max(axis=1)]).T
                # delta=reward + gamma * np.max(np.dot(X,w))-np.dot(x,w)
                # w=w+ alpha*delta[0]* x.T

                obs=new_obs
                r=r+reward
                judge=(perFeature==x).all(1)
                if not judge.all() or m==1:#(x.tolist() not in perFeature):
                    # perFeature.append(x.tolist())
                    # perReward.append(0)
                    # T.append(-1)
                    perFeature=np.r_[perFeature, x]
                    perReward=np.r_[perReward,np.zeros([1,1])]
                    T=np.r_[T,np.zeros([1,1])]
                # T = [i+1 for i in T]
                # tmp=[pow(gamma,i) for i in T ]
                # perReward=np.array(perReward)+np.dot(reward,tmp)
                # perReward=perReward.tolist()
                T=T+1
                perReward=perReward+np.dot(reward,pow(gamma,T))
                # if m==10:
                #     break
            elif info is not None:
                print("round result: own hp {} vs opp hp {}, you {}".format(info[0], info[1],
                                                                            'win' if info[0]>info[1] else 'lose'),'训练局数',n)
                if info[0]>info[1]:
                    NumWin=NumWin+1
            else:
                # java terminates unexpectedly
                pass
        # delta1=(np.array(perReward)-np.dot(perFeature,w))
        delta1=(perReward-np.dot(perFeature,w))
        for i in range(len(delta1)):
            perFeature[i,:]=delta1[i,0]*perFeature[i,:]
        delta=np.sum(perFeature,axis=0)
        w=w.T+alpha*delta
        w=w.T
        RewardData.append(r)
        if n<N:
            if np.mod(n,2)==0:
                string="./trianLog/MCtrainLog/w"+str(n+p)
                io.savemat(string, {'w': w})
                string="./trianLog/MCtrainLog/reward"+str(n+p)
                io.savemat(string, {'r': RewardData})
        else:
            print("训练结束")
            string="./trianLog/MCtrainLog/w"+str(n+p)
            io.savemat(string, {'w': w})
            string="./trianLog/MCtrainLog/reward"+str(n+p)
            io.savemat(string, {'r': RewardData})
            break

    print("finish training")
    print("获胜局数",NumWin)
