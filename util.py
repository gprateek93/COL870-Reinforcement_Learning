from RL_algorithms import monte_carlo,td_learning,n_step_sarsa,q_learning,sarsa_lambda
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def visualise(Q,title="",dir_name=""):
    if not os.path.exists(dir_name):
        print("making ",dir_name)
        name= dir_name.split('/')
        for n in name:
            path = os.path.join(".",n)
            if not os.path.exists(path):
                os.mkdir(n)
            os.chdir(n)
    
    indicator_dict = {}

    for t1 in range(0,3):
        for t2 in range(0,3):
            for t3 in range(0,3):
                t = (t1,t2,t3)
            
                indicator_dict[t] = {
                    1:{"x":[],"y":[],"q":[]},
                    0:{"x":[],"y":[],"q":[]}
                }
                
    for k, v in Q.items():
        playerSum = k[0][0]
        dealercard = k[0][1][0]
        indicator_state = k[0][2]
        action=k[1]
        
       
        indicator_dict[indicator_state][action]['x'].append(playerSum)
        indicator_dict[indicator_state][action]['y'].append(dealercard)
        indicator_dict[indicator_state][action]['q'].append(v)
        
    i = 0
    for k,v in indicator_dict.items():
        for a in (0,1):
            x = indicator_dict[k][a]["x"]
            y = indicator_dict[k][a]["y"]
            z = indicator_dict[k][a]["q"]
            fig = plt.figure(figsize=[15, 6])
        
            ax = fig.gca(projection='3d')
            surf = ax.plot_trisurf(x,y,z, cmap=plt.cm.coolwarm)
            #ax.scatter(x, y, z)
            ax.set_title(str(k) +"Action :"+ str(a))
            ax.set_xlabel("player sum")
            ax.set_ylabel("dealer showing")
            ax.set_zlabel("reward")
            filename = str(i)+".jpg"
            plt.savefig(filename)
            i+=1
    print("done")

def performance(total_experiments = 10):
    alpha = 0.1
    gamma = 1.0
    epsilon = 0.1
    episodes = 100
    k = [1, 10, 100, 1000]
    for n in k:
        print("k=",n)
        plt.figure(figsize=(15, 4))
        avg1 = [0 for e in range(episodes)]
        avg2 = [0 for e in range(episodes)]
        avg3 = [0 for e in range(episodes)]
        avg4 = [0 for e in range(episodes)]
        for e in range(total_experiments):
            print("Experiment",e)
            _, t1 = n_step_sarsa(episodes,alpha,gamma,n,epsilon)
            _, t2 = n_step_sarsa(episodes,alpha,gamma,n,epsilon,decay=True)
            _, t3 = q_learning(episodes,alpha,gamma,epsilon,n)
            _, t4 = sarsa_lambda(episodes,alpha,gamma,epsilon)
            avg1= [avg1[i] + t1[i]/total_experiments for i in range(episodes)] #running average
            avg2= [avg2[i] + t2[i]/total_experiments for i in range(episodes)]
            avg3= [avg3[i] + t3[i]/total_experiments for i in range(episodes)]
            avg4= [avg4[i] + t4[i]/total_experiments for i in range(episodes)]
        plt.plot(avg1,label="k="+str(n)+"method = n_step_sarsa")
        plt.plot(avg2,label="k="+str(n)+"method = n_step_sarsa_with_decay")
        plt.plot(avg3,label="k="+str(n)+"method = q-learning")
        plt.plot(avg4,label="k="+str(n)+"method = sarsa_lambda")
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.savefig(str(n)+".jpg")
        plt.legend()
        plt.show()

def getAllGraphs():
    #monte carlo for episodes = 100 and 10000 both for early visit and first visit
    Q1 = monte_carlo(100,mode = "fv")
    Q2 = monte_carlo(10000,mode = "fv")
    Q3 = monte_carlo(100,mode = "ev")
    Q4 = monte_carlo(10000,mode = "ev")
    os.chdir("../..")
    visualise(Q1,dir_name="MC/FV/100")
    os.chdir("../..")
    visualise(Q2,dir_name="MC/FV/10000")
    os.chdir("../..")
    visualise(Q3,dir_name="MC/EV/100")
    os.chdir("../..")
    visualise(Q4,dir_name="MC/EV/10000")
    os.chdir("../..")
    #td for k = 1,3,5,10,10,1000 for 100 and 10000 for each
    Q5 = td_learning(100,k=1)
    Q6 = td_learning(10000,k=1)
    Q7 = td_learning(100,k=3)
    Q8 = td_learning(10000,k=3)
    Q9 = td_learning(100,k=5)
    Q10 = td_learning(10000,k=5)
    Q11 = td_learning(100,k=10)
    Q12 = td_learning(10000,k=10)
    Q13 = td_learning(100,k=100)
    Q14 = td_learning(10000,k=100)
    Q15 = td_learning(100,k=1000)
    Q16 = td_learning(10000,k=1000)
    visualise(Q5,dir_name="TD/1/100")
    os.chdir("../..")
    visualise(Q6,dir_name="TD/1/10000")
    os.chdir("../..")
    visualise(Q7,dir_name="TD/3/100")
    os.chdir("../..")
    visualise(Q8,dir_name="TD/3/10000")
    os.chdir("../..")
    visualise(Q9,dir_name="TD/5/100")
    os.chdir("../..")
    visualise(Q10,dir_name="TD/5/10000")
    os.chdir("../..")
    visualise(Q11,dir_name="TD/10/100")
    os.chdir("../..")
    visualise(Q12,dir_name="TD/10/10000")
    os.chdir("../..")
    visualise(Q13,dir_name="TD/100/100")
    os.chdir("../..")
    visualise(Q14,dir_name="TD/100/10000")
    os.chdir("../..")
    visualise(Q15,dir_name="TD/1000/100")
    os.chdir("../..")
    visualise(Q16,dir_name="TD/1000/10000")
    os.chdir("../..")

# getAllGraphs()
# q,_ = n_step_sarsa(10000,0.1,1,1000,0.1)
# visualise(q,dir_name="sarsa")
performance()