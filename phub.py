import numpy as np
class phub_population(object):
    def __init__(self,data, n_nodes=25, n_hubs=4) -> None:
        #numero de nodos
        self.n_nodes = n_nodes
        #numero de hubs
        self.n_hubs = n_hubs
        #numero de vecinos(necesario para ga)
        self.neigs = 3

        #leemos los datos
        self.data, self.shortes_ways = data
        #hubs iniciales
        self.hubs = np.random.choice(np.arange(n_nodes), n_hubs, replace=False)
        #conexiones nodo-hub
        self.allocation = [0 for _ in range(self.n_nodes)]
        self.generate_allocation()

    def calculate_fitness(self, alpha = 0.75):
        """Se calcula el fitness de la posible solución como la suma de costes para todos los recorridos.
        Puesto que se trata de minimizar la suma de costes, se devulve su inverso."""
        
        fitness = 0

        #distancia de cada nodo a su hub, las distancias se dividen entre 1000
        fitness_list = np.array([self.data[i][al][0]/1000 * self.data[i][al][1] for i,al in enumerate(self.allocation)])
        #se multiplica la distancia de cada nodo a su hub el número de veces que se puede realizar y se suman
        fitness+=np.sum(2*(self.n_nodes-1)*fitness_list) 

        #calculo de recorrido entre hubs
        for h in self.hubs: #se recorren los hubs
            #numero de nodos no hubs que estan unidos al hub h
            mult = np.count_nonzero(self.allocation==h)
            #se calcula la distancia entre el hub h y el resto de hubs, y se multiplica por el numero de veces que se vaya a realizar ese recorrido
            tmp = np.array([self.data[h][h_][0]/1000 * self.data[h][h_][1] * alpha * (mult * np.count_nonzero(self.allocation==h_)) for h_ in self.hubs])
            fitness+=np.sum(tmp)
        
        #se devuelve el inverso, ya que se quiere minimizar los costes
        return 1/fitness
    
    def generate_allocation(self):
        """Generación de la lista que contiene cual es el hub más cercano para cada nodo"""
        #se recorren la lista de lista que contienen los nodos mas cercanos para cada nodo i
        for en,i in enumerate(self.shortes_ways[:self.n_nodes]):
            j=0
            #para cada nodo i, asignamos que se una a su nodo que tenga menor coste
            while i[j] not in self.hubs:
                j+=1
            self.allocation[en] = i[j]
        #recalculamos el fitness
        self.fitness = self.calculate_fitness()

    def copy(self):
        """Devuelve un nuevo objeto con los mismos hubs que el actual"""
        #se crea el nuevo objeto
        temp = phub_population((self.data,self.shortes_ways), n_hubs=self.n_hubs)
        #se asignan los hubs
        temp.hubs = self.hubs.copy()
        
        return temp

    def generate_neigh(self):
        """Cambio aleatoriamente un hub a uno de sus nodos más cercanos que no sea hub"""
        #creamos una poblocion tmp que va a ser la nueva población
        temp = phub_population((self.data,self.shortes_ways), n_hubs=self.n_hubs)
        #inicialmente tiene los mismos hubs
        temp.hubs = self.hubs.copy()
        #eligimos el hub que se va a modificar aleatoriamente
        hub_change_index = np.random.choice(self.n_hubs)
        #cojemos la distancia desde el hub seleccionado al resto de nodos
        neigs_hub = self.shortes_ways[temp.hubs[hub_change_index]]

        #eliminamos los nodos que son hubs (puesto que si saliera uno de estos disminuiria el número total de hubs)
        for tb in temp.hubs:
            neigs_hub = np.delete(neigs_hub, np.where(neigs_hub == tb))

        #cojemos los n(3) vecinos más cercanos
        neigs_hub_correted = neigs_hub[:self.neigs]

        #actualizamos el valor del hub como una selección aleatoria de entre los vecinos (n nodos mas cercanos)
        temp.hubs[hub_change_index] = np.random.choice(neigs_hub_correted)
        #recalculamos allocation (asociaciones nodo-hub) y fitness
        temp.generate_allocation()
        temp.fitness = temp.calculate_fitness() 

        return temp

def read_txt(file="phub.txt"):
    """
    Genera una lista con el distancia(w) y trafico(c) de cada nodo
    [[[nw11,nc11][nw12,nc12] ... [nw1N, nc1N]]
     [[nwN1,ncN1]    ...     ... [nwNN, ncNN]]]

    """
    #creamos la lista de listas que se va a devolver
    data_processed = [[] for _ in range(25)]
    #leemos datos del txt
    f = open(file, "r")    

    #recorremos los datos leidos
    for a in (f.read().splitlines()):
        a_=a.split()
        #guardamos en la lista los dos ultimos elemenotos leidos (distancia, trafico)
        data_processed[int(a_[0])-1].append(a_[-2:])
    data_processed = np.array(data_processed, dtype=float)

    #creamos una lista de listas en donde se ordenan los nodos según sus costes(de menor a mayor). Pe. [[1,3],[3,1]] significa que el nodo más cercano del nodo 1 es el 1, y del nodo 2 el 3.
    shortest_ways = [(arr*arr2).argsort() for arr,arr2 in zip(data_processed[:,:,1],data_processed[:,:,0])]

    return data_processed, shortest_ways

if __name__ == "__main__":
    data = read_txt()
    #ph=phub_population(data)
    #print(1/ph.calculate_fitness())