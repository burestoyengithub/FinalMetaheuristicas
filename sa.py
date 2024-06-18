from phub import *

class sa():
    def __init__(self, n_phubs=4, max_int=10000, L=80) -> None:
        #cargamos los datos
        self.data = read_txt("phub.txt")
        #cargamos la solucion inicial
        self.ph   = phub_population(self.data, n_hubs=n_phubs)
        #cargamos la temperatura inicial
        self.T_0 = np.abs((1/self.ph.fitness - 1/phub_population(self.data, n_hubs=n_phubs).fitness))
        #la temperatura minima es inalcanzable para fijar el rendimiento por epocas
        self.Tmin = -0.01
        #valor de decaida de temperatura
        self.alpha = 0.90                                      
        #ejecuciones con la misma temperatura (proporcional al número de vecinos y al espacio de busqueda)
        self.L = L #2**self.ph.neigs * self.ph.n_hubs                    
        #numero maximo de iteraciones globales
        self.max_int = max_int

        self.avg_list=[]
        self.min_list=[np.inf]
        self.max_list=[0]

    def train(self, verbose=False):
        """Entrena sa"""
        maximo = 0
        x=0
        T=self.T_0

        e_list=[]
        e_list_index=[]
        #bucle principal
        while T>self.Tmin and x<self.max_int:
            #bucle con la misma temperatura
            for _ in range(self.L):
                #fitness de la solucion actual
                fitness = self.ph.fitness
                #vecino de la solucion actual (nueva solucion)
                new_ph = self.ph.generate_neigh()
                #fitness de la nueva posible solucion
                fitness_new = new_ph.fitness

                #calculamos si se acepta:
                #se acepta simpre si es mejor
                #si no lo es se calcula una probablidad depediente de lo buena que sea la nueva solucion y del valor de temperatura
                acepted = True if fitness_new>fitness else np.e**((1/fitness - 1/fitness_new)/T) > np.random.rand()

                #si se acepta se asigna como solucion a la nueva solucion
                if acepted: self.ph=new_ph
                #guardamos el valor máximo para analisis
                if self.ph.fitness>maximo: maximo=self.ph.fitness
                #interacciones
                x+=1
                #guardamos valores de la probablidad para analisiss
                if fitness_new<fitness:
                    e_list.append(np.e**((1/fitness - 1/fitness_new)/T))
                    e_list_index.append(x)

                self.avg_list.append(1/self.ph.fitness)
                self.min_list.append(min(self.min_list[-1],self.avg_list[-1]))
                self.max_list.append(max(self.max_list[-1],self.avg_list[-1]))
               
                #datos por pantalla
                if verbose:print(f'Round {x}, temp {T:.2f} actual sol fit {self.avg_list[-1]:.2f} better sol fit {self.min_list[-1]:.2f}')
                #si se llega al maximo de interacciones se acaba
                if x>self.max_int:break
            #actualizacion de la temperatura
            T = self.alpha*T

        return self.avg_list, self.min_list[1:], self.max_list[1:], e_list, e_list_index
    
if __name__ == "__main__":
    sa_=sa(n_phubs=4)
    sa_.train(verbose=True)