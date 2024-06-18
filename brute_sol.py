from phub import *
from itertools import combinations


def get_ft(p):
    """Devuelve el fitness de una poblacion"""
    ph = phub_population(data,n_nodes=n_nodes, n_hubs=len(p))
    ph.hubs = p
    ph.generate_allocation()
    return ph.fitness

#numero de nodos
n_nodes = 25
#numero de hubs (p)
n_hubs =  5
#se abre el fichero para guardar los resultados
f=open("best_results.txt","a")

#rango en el que se calculan los los hubs
for n_hubs in range(1 ,26):

    #se genera un array con todas las posibles combinaciones de nodos/hubs
    a = np.arange(n_nodes)
    posibilities = (np.array(list(combinations(a, n_hubs))))

    #datos
    data= read_txt("phub.txt")

    #se calcula para cada posibilidad el fitness
    dict = {str(p):get_ft(p) for p in np.array(posibilities) }

    #se guarda el mejor resultado (mayor valor)
    p_opt = (max(dict, key=dict.get))

    #se muestran datos por pantalla
    print(n_hubs)
    print(p_opt)
    print(1/dict[p_opt])
    print()

    #escribe los resultados
    f.write(f'{n_hubs} \t {p_opt[1:-1]} \t {1/dict[p_opt]:.2f} \n') 

f.close()

