'''
2020/6/01
Tong Ru
test —— traditional MC
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


    w= io.loadmat("./trainedWeight/w736-MC")#使用训练好的权重测试
    w=w['w']

    n=0
    Ntest=50#测试局数
    NumWin=0
    score=0
    RewardData=[]

    A=np.zeros((40,144*40))
    act = random.randint(0, 39)#初始动作
    for i in range(0,40):
        A[i,i*144:(i+1)*144]=1

    while True:
        obs = env.reset(env_args=env_args)
        reward, done, info = 0, False, None
        n=n+1#局数+1
        r=0

        while not done:
            # TODO: or you can design with your RL algorithm to choose action [act] according to game state [obs]
            X=np.tile(obs,(40,40))
            X=X*A
            X=X / np.array([X.max(axis=1)]).T
            act=np.argmax(np.dot(X,w))

            # x=np.zeros((1,144*40))
            # x[0,144*act:144*(act+1)]=np.array([obs])
            new_obs, reward, done, info = env.step(act)

            if not done:
                # TODO: (main part) learn with data (obs, act, reward, new_obs)

                obs=new_obs
                r=r+reward
                # print(reward,'比分',info[0],info[1])

            elif info is not None:
                print("round result: own hp {} vs opp hp {}, you {}".format(info[0], info[1],
                                                                            'win' if info[0]>info[1] else 'lose'),'测试局数',n)
                print('last reward',reward)
                if info[0]>info[1]:
                    NumWin=NumWin+1
                    score=score+(info[0]-info[1])
            else:
                # java terminates unexpectedly
                pass

        RewardData.append(r)
        if n==Ntest:
            if NumWin==0:
                meanscore=0
            else:
                meanscore=score/NumWin

            print("获胜局数",NumWin)
            print("平均得分差距",meanscore)
            string="./trianLog/MCtrainLog/reward-test"
            io.savemat(string, {'r': RewardData})
            break


    print("finish training")
