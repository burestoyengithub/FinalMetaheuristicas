from phub import *

class ga():
    def __init__(self, n_phubs, max_int=10000) -> None:
        #longitud de los genes (el número de hubs)
        self.gl         = n_phubs                 
        #longitud de la poblacion     
        self.popsize    = 512    
        #Posibilidad de cruce
        self.pc         = 0.8
        #probabilidad de mutacino
        self.pm         = 0.01
        #Numero de ejecuciones del bucle
        self.MAX_ISTEPS = max_int                         
        #data
        self.data = read_txt("phub.txt")
        self.setup_population()

        self.avg_list=[]
        self.min_list=[]
        self.max_list=[]

    def setup_population(self):
        """Creacion de la poblacion"""
        #se crea un objeto phub_population para cada posible solucion con una inicializacion aleatoria
        self.population_list = [phub_population(self.data, n_hubs=self.gl) for _ in range(self.popsize)]
        self.population = {p: p.fitness for p in self.population_list}
        

    def train(self,verbose=False):
        """Se entrena el ga"""
        for i in range(self.MAX_ISTEPS):
            #seleccion de padres
            p1, p2 = self.select_tournament(),self.select_tournament()
            #hijo por cruce
            p = self.single_point_crossover(p1,p2)
            #posible mutacion
            p = self.mutate(p)
            #actualización de parametros internos de la clase
            p.generate_allocation()
            #se introduce el hijo en la poblacion
            self.population = self.replacement(p)
            
            #guardamos los datos para analisis
            self.avg_list.append(1/np.mean(list(self.population.values())))
            self.min_list.append(1/np.max(list(self.population.values())))
            self.max_list.append(1/np.min(list(self.population.values())))
            
            #datos por pantalla
            if verbose:print(f'Round {i}, avg fitness {1/np.mean(list(self.population.values())):.2f} min dist {1/np.max(list(self.population.values())):.2f} max dist {1/np.min(list(self.population.values())):.2f}')
      
        return self.avg_list, self.min_list, self.max_list
            
    def select_tournament(self):
        """Se calcula la selecion por torneo sobre toda la pobracion"""
        #se eligen dos individuos de toda la poblacion aleatoriamente
        [p1,p2] = np.random.choice(list(self.population.keys()),2, replace=False) 
        #devolvemos el que tiene mayor valor de fitness
        return p1 if p1.fitness>p2.fitness else p2

    def single_point_crossover(self, p1, p2):
        """Se calcula el cruce por un punto"""
        #no se ejecuta si se genera un número mayor a la probabilidad
        if np.random.rand() > self.pc:
            #en ese caso se devuleve uno de los padres aleatoriamente
            return np.random.choice([p1,p2]).copy() 
        
        hubs_tmp=[]
        p1.hubs = np.short(p1.hubs)
        p2.hubs = np.short(p2.hubs)
        #dado que el hijo no puede tener menor numero de hubs que los padres (no se pueden repetir los hubs), se calculan puntos aleatorio hasta que se satisfaga esa condicion
        while len(np.unique(hubs_tmp))<p1.n_hubs:
            #creamos el punto
            crossover_point = np.random.choice(self.gl+1)
            #creamos el hijo
            hubs_tmp = np.concatenate((p1.hubs[:crossover_point], p2.hubs[crossover_point:]))
            #si el hijo no es valido comprobamos el otro hijo generado
            hubs_tmp = hubs_tmp if len(np.unique(hubs_tmp))<p1.n_hubs else np.concatenate((p2.hubs[:crossover_point], p1.hubs[crossover_point:]))
        
        #cremos un nuevo individuo, le cargamos los hubs y lo devolvemos
        p = phub_population(self.data,n_hubs=self.gl)
        p.hubs = hubs_tmp

        return p

    def mutate(self, p):
        """Se calcula la mutación"""
        #Se calcula que genes mutan aleatoriamente
        mutations = np.random.rand(len(p.hubs)) <= self.pm     
        #si no hay mutaciones en ningun elemento se devuelve el individuo igual
        if (mutations==False).all(): return p
    
        #para cada posible mutacion calculamos a que valores puede mutar. No puede mutar a un gen que ya este, porque disminuiria el numero de hubs (se repetirian)
        posible_mutations = np.setdiff1d(np.arange(p.n_nodes), p.hubs[np.logical_not(mutations)])
        new_hubs_mutated = np.random.choice(posible_mutations, np.count_nonzero(mutations), replace=False)
        p.hubs[mutations] = new_hubs_mutated

        return p

    def replacement(self,p):
        """Se realiza el remplazo"""
        #se calcula cual es el peor individuo de la poblacion
        worst_p = np.argmin(list(self.population.values()))
        #lo eliminamos
        self.population.pop(list(self.population.keys())[worst_p])
        #introducimos el nuevo individuo
        self.population[p] = p.fitness  

        return self.population
    
if __name__ == "__main__":
    ga = ga(n_phubs=4)
    ga.train(verbose=True)