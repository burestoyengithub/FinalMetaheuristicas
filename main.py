from uu import Error
from bs4 import BeautifulStoneSoup
from regex import F, P
from ga import *
from sa import *
import matplotlib.pyplot as plt
import math

def get_data(range_hubs,name='exp/', rounds=10, max_int=5000):
    """Genera los datos para los entrenamientos de sa y ga para los hubs en range_hubs
    ejecutandolos rounds veces para disminuir la componente estocastica"""

    #recorre el todos los hubs
    for i in (range_hubs):
        print(f'For {i} hubs')
        #carga los ficheros donde se guardan las soluciones
        f_sa_min = open(name + f'exp_sa_min_{i}_hubs.txt',"w")
        f_ga_min = open(name + f'exp_ga_min_{i}_hubs.txt',"w")

        f_sa_avg = open(name + f'exp_sa_avg_{i}_hubs.txt',"w")
        f_ga_avg = open(name + f'exp_ga_avg_{i}_hubs.txt',"w")

        f_sa_max = open(name + f'exp_sa_max_{i}_hubs.txt',"w")
        f_ga_max = open(name + f'exp_ga_max_{i}_hubs.txt',"w")

        elist_f = open(name + f'elist{i}_hubs.txt',"w")
        index_f = open(name + f'elist_index{i}_hubs.txt',"w")

        #recorre el numero de ejecuciones para eliminar la componente estocastica
        for _ in range(rounds):
            print(f'\t Round {_}')
            #entrenamiento sa
            sa_=sa(n_phubs=i, max_int=max_int)
            avg_list_sa, min_list_sa, max_list_sa ,elist, index = sa_.train(verbose=False)
            #se guardan los datos
            save_data(f_sa_min, min_list_sa)
            save_data(f_sa_avg, avg_list_sa)
            save_data(f_sa_max, max_list_sa)
            save_data(elist_f, elist)
            save_data(index_f, index)

            #entrenamiento  ga
            ga_ = ga(n_phubs=i, max_int=max_int)
            avg_list_ga, min_list_ga, max_list_ga = ga_.train(verbose=False)
            #se guardan los datos
            save_data(f_ga_min, min_list_ga)
            save_data(f_ga_avg, avg_list_ga)
            save_data(f_ga_max, max_list_ga)


        print()
        f_sa_min.close()
        f_ga_min.close()
        f_sa_avg.close()
        f_ga_avg.close()

def make_all_table(path_sa_min, path_ga_min, ga_sa_term, path_brute):
    """Se genera la tabla en formato latex que aparece en la memoria"""

    sa_sols = []
    sa_sols_variance = []
    ga_sols = []
    ga_sols_variance = []
    sa_t    = []
    sa_t_variance = []
    ga_t    = []
    ga_t_variance = []

    f_brute = open(path_brute, "r")
    f_brute_lines=f_brute.read().split("\n")[1:-1]
    f_brute_complete = [np.array(f_c.split(), dtype=float)[-1] for f_c in f_brute_lines]

    for i in range(1,25):
        brute = f_brute_complete[i-1]/1000

        f_sa = open(path_sa_min+ str(i) +ga_sa_term,"r")
        f_sa_lines=f_sa.read().split("\n")[:-1]
        f_sa_complete = np.array([[min(np.array(f_sa_line.split(),dtype=float)),np.argmin(np.array(f_sa_line.split(),dtype=float))] for f_sa_line in f_sa_lines])
        min_sa   =    np.mean(f_sa_complete[:,0]/1000)
        std_min_sa   = np.std(f_sa_complete[:,0]/1000) 
        t_min_sa =    np.mean(f_sa_complete[:,1])
        std_t_min_sa = np.std(f_sa_complete[:,1])
        

        f_ga = open(path_ga_min+ str(i) + ga_sa_term,"r")
        f_ga_lines=f_ga.read().split("\n")[:-1]
        f_ga_complete = np.array([[min(np.array(f_ga_line.split(),dtype=float)),np.argmin(np.array(f_ga_line.split(),dtype=float))] for f_ga_line in f_ga_lines])
        min_ga   =    np.mean(f_ga_complete[:,0]/1000) 
        std_min_ga   = np.std(f_ga_complete[:,0]/1000) 
        t_min_ga =    np.mean(f_ga_complete[:,1])
        std_t_min_ga = np.std(f_ga_complete[:,1])


        txt = ("{n_hubs:d} & {brute_:.2f} & {min_ga_:.2f} {{\scriptsize({std_ga_:.2f})}} & {min_t_ga_:.2f} {{\scriptsize({std_t_ga_:.2f})}} & {min_sa_:.2f} {{\scriptsize({std_sa_:.2f})}} & {min_t_sa_:.2f} {{\scriptsize({std_t_sa_:.2f})}} \\\\")
        print(txt.format(n_hubs=i, brute_=brute, min_ga_= min_ga, std_ga_ = std_min_ga, min_t_ga_= t_min_ga, std_t_ga_ = std_t_min_ga,\
                                                                        min_sa_= min_sa, std_sa_ = std_min_sa, min_t_sa_= t_min_sa, std_t_sa_ = std_t_min_sa   ))

        sa_sols.append(min_sa/brute)
        sa_sols_variance.append(std_min_sa/brute)
        ga_sols.append(min_ga/brute)
        ga_sols_variance.append(std_min_ga/brute)

        sa_t.append(t_min_sa)
        sa_t_variance.append(std_t_min_sa)
        ga_t.append(t_min_ga)
        ga_t_variance.append(std_t_min_ga)

    print(sa_sols_variance)
    fig, ax = plt.subplots()
    legend=["SA solutions",
            "SA variance",
            "GA solutions",
            "GA variance"]
    plot_stuff(ax, False, legend,"green", sa_sols)
    plot_stuff(ax, True, legend, "green", np.array(sa_sols)-np.array(sa_sols_variance), np.array(sa_sols)+np.array(sa_sols_variance))
    plot_stuff(ax, False, legend,"blue", ga_sols)
    plot_stuff(ax, True, legend, "blue", np.array(ga_sols)-np.array(ga_sols_variance), np.array(ga_sols)+np.array(ga_sols_variance))
    ax.set_ylim(0.95,1.05)
    ax.set_xlabel('Number of hubs',fontsize=20)
    ax.set_ylabel('Similarity to optimous solution',fontsize=20)   
    plt.legend(legend,fontsize=20)

    fig, ax1 = plt.subplots()
    legend=["SA time solutions",
            "SA time variance",
            "GA time solutions",
            "GA time variance"]
    plot_stuff(ax1, False, legend,"green", sa_t)
    plot_stuff(ax1, True, legend, "green", np.array(sa_t)-np.array(sa_t_variance), np.array(sa_t)+np.array(sa_t_variance))
    plot_stuff(ax1, False, legend,"blue", ga_t)
    plot_stuff(ax1, True, legend, "blue", np.array(ga_t)-np.array(ga_t_variance), np.array(ga_t)+np.array(ga_t_variance))

    ax1.set_ylim(0, 5000)

    plt.xlabel('Number of hubs',fontsize=20)
    plt.ylabel('Rounds to reach the solution',fontsize=20)   
    plt.legend(legend,fontsize=20)

    #ax.set_ylim(min(min_list_ga[-1], min_list_ga[-1]), 3*min(min_list_ga[-1], min_list_ga[-1]))
    plt.show()

def main(n_hubs):
    """Se realiza un entrenamiento sobre n hubs utilizando recocido simulado
    y algoritmos genéticos y se muestran los resultados en graficas"""
    #numero de hubs
    hubs = n_hubs
    #maximo numero de interacciones
    max_int = 5000
    
    #clase del recocido simulado
    sa_=sa(n_phubs=hubs, max_int=max_int)
    #se entrena
    avg_list_sa, min_list_sa, max_list_sa ,elist, index = sa_.train(verbose=False)

    #clase del algoritmo genetico
    ga_ = ga(n_phubs=hubs, max_int=max_int)
    #se entrena
    avg_list_ga, min_list_ga, max_list_ga = ga_.train(verbose=False)

    #se muestran los resultados por pantalla
    print(f' sol sa {min_list_sa[-1]:.2f} {np.argmin(min_list_sa)} sol ga {min_list_ga[-1]:.2f} {np.argmin(min_list_ga)}')

    #se generan gráficas de valores minimos y maximos
    fig, ax1 = plt.subplots()
    legend=["SA max-min",
           "SA current val",
           "GA max-min",
           "GA mean val"]
    plot_stuff(ax1, True, legend, "green", min_list_sa, max_list_sa)
    plot_stuff(ax1, False, legend,"green", avg_list_sa)
            
    plot_stuff(ax1, True, legend, "blue", min_list_ga, max_list_ga)
    plot_stuff(ax1, False, legend,"blue", avg_list_ga)
    ax1.set_ylim(min(min(min_list_ga), min(min_list_sa)) , max(max(max_list_ga), max(max_list_sa)*0.8))

    #se generan gráficas de valores de e
    fig, ax2 = plt.subplots()
    ax2.plot(index,elist)


    #se generan gráficas de la solucion
    fig, ax = plt.subplots()
    legend=["SA sol",
            "GA sol"]
    plot_stuff(ax, False, legend,"green", min_list_sa)
    plot_stuff(ax, False, legend,"blue", min_list_ga)
    

    plt.xlabel('Round',fontsize=20)
    plt.ylabel('Fitness',fontsize=20)   
    plt.legend(legend,fontsize=8)
    #ax.set_ylim(min(min_list_ga[-1], min_list_ga[-1]), 3*min(min_list_ga[-1], min_list_ga[-1]))
    plt.show()
    
def read_and_plot(path_sa, path_ga, path_brute, hubs, path):
    """Lee los resultados de un entrenamietno y genera gráficas para ellos"""
    
    n_hubs = '_' + hubs + '_hubs.txt'
    
    f_brute = open(path_brute, "r")
    f_brute_lines=f_brute.read().split("\n")[1:-1]
    f_brute_complete = [np.array(f_c.split(), dtype=float)[-1] for f_c in f_brute_lines]
    brute_sol = f_brute_complete[int(hubs)-1]/1000

    f_sa_min = open(path_sa + 'min' + n_hubs,"r")
    f_sa_max = open(path_sa + 'max' + n_hubs,"r")
    f_sa_avg = open(path_sa + 'avg' + n_hubs,"r")

    f_ga_min = open(path_ga + 'min' + n_hubs,"r")
    f_ga_max = open(path_ga + 'max' + n_hubs,"r")
    f_ga_avg = open(path_ga + 'avg' + n_hubs,"r")

    f_elist = open(path + f'elist{n_hubs[1:]}',"r")
    f_index = open(path + f'elist_index{n_hubs[1:]}',"r")
    
    #se transoforman los datos a arrays
    sa_min = np.array([sa_line.split()     for sa_line     in f_sa_min.read().split("\n")[:-1]], dtype=float)[2]/1000#, axis=0)
    sa_max = np.array([sa_line_max.split() for sa_line_max in f_sa_max.read().split("\n")[:-1]], dtype=float)[2]/1000#, axis=0)
    sa_avg = np.array([sa_line_avg.split() for sa_line_avg in f_sa_avg.read().split("\n")[:-1]], dtype=float)[2]/1000#, axis=0)
    ga_min = np.array([ga_line.split()     for ga_line     in f_ga_min.read().split("\n")[:-1]], dtype=float)[2]/1000#, axis=0)
    ga_max = np.array([ga_line_max.split() for ga_line_max in f_ga_max.read().split("\n")[:-1]], dtype=float)[2]/1000#, axis=0)
    ga_avg = np.array([ga_line_avg.split() for ga_line_avg in f_ga_avg.read().split("\n")[:-1]], dtype=float)[2]/1000#, axis=0)
    
    #calculamos los valores de e
    base = np.ones((10,5010))*0
    base_2 = np.ones(5010)*0
    for e,(elist_line,index_line) in enumerate(zip(f_elist.read().split("\n")[:-1],f_index.read().split("\n")[:-1])):
        base[e][np.array(index_line.split(), dtype=int)] = np.array(elist_line.split())
        base_2[np.array(index_line.split(), dtype=int)] += 1
    data_e = np.sum(base, axis=0)
    data_e[base_2!=0] /= base_2[base_2!=0]


    #se generan gráficas de la soucion
    fig, ax1 = plt.subplots()
    legend=["Current value of SA",
            "Minimum and maximun SA values found",
            "Average value of GA population",
            "Minimum and maximun GA values in population",
            "Optimous solution"]
    plot_stuff(ax1, False, legend,"green", sa_avg)
    plot_stuff(ax1, True, legend, "green", sa_min, sa_max)
            
    plot_stuff(ax1, False, legend,"blue", ga_avg)
    plot_stuff(ax1, True, legend, "blue", ga_min, ga_max)
    plot_stuff(ax1, False, legend, "black",  np.ones(len(ga_avg))*brute_sol, wt=3)
    ax1.set_ylim(min(ga_min[-1], sa_min[-1])*0.95, max(max(ga_max), max(sa_max))*1.05)
    plt.legend(legend,fontsize=20)
    
    #se generan gráficas de valores de e
    fig, ax2 = plt.subplots()
    ax2.plot(np.where(base_2!=0)[0],data_e[base_2!=0])

    #se generan gráficas de la solucion
    fig, ax = plt.subplots()
    legend=["SA solution",
            "GA solution",
            "Optimous solution"]
    plot_stuff(ax, False, legend,"green", sa_min)
    plot_stuff(ax, False, legend,"blue", ga_min)
    plot_stuff(ax, False, legend, "black",  np.ones(len(ga_avg))*brute_sol)
    ax.set_ylim(min(ga_min[-1], sa_min[-1])*0.99, max(max(ga_min), max(sa_min))*0.7)

    ax1.set_xlabel('Round',fontsize=20)
    ax1.set_ylabel('Cost',fontsize=20)   

    ax2.set_xlabel('Round',fontsize=20)   
    ax2.set_ylabel('Probability',fontsize=20)   

    plt.xlabel('Round',fontsize=20)
    plt.ylabel('Cost',fontsize=20)   
    plt.legend(legend,fontsize=20)

    plt.show()

def plot_stuff(ax, fill, legend, color,  x, y=None, wt=2):
    """Funcion para dibujar los graficos"""
    if fill:
        ax.fill_between(np.arange(len(x)), x, y, alpha=.3, linewidth=0, color="dark"+color)
    else:
        ax.plot(x, color=color, marker="",linewidth=wt)

    ax.legend(legend)

def save_data(f,lista):
    """Funcion para guardar los datos"""
    for l in lista:
        f.write((str(l)+" "))
    f.write("\n")

if __name__ == "__main__":
    #get_data(range(1,25),name="exp_4/", rounds=10)
    main(8)
    
    make_all_table("exp/exp_sa_min_","exp/exp_ga_min_","_hubs.txt","best_results2.txt")

    read_and_plot("exp/exp_sa_","exp/exp_ga_","best_results2.txt", '8', path='exp/')

