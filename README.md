# Comparación de un algoritmo poblacional y uno de trayectoria (p-hub)
### Resolución de problemas con metaheurísticos
### José Miguel Burés

## Programas
__brute_sol.py__ Calcula las soluciones óptimas para el problema utilizando un algoritmo de fuerza bruta

__ga.py__ Adaptación del programa proporcionado en Java para el algoritmo genetico. Se utiliza selección por torneo y cruce por punto.
    
__main.py__ Es donde se guardan las funciones principales (más detalle e Ejecución).

## Archivos complementarios
    best_results2.txt: Soluciones calculadas de brute_sol.py
    phub.txt: Datos del problema
    exp: Carpeta con los resultados de la bateria de entrenamientos

## Ejecución
En main.py hay 4 posibles ejecuciones:
    
    - get_data((n,N),name, rounds): Realiza una bateria de *rounds* ejecuciones y guarda los resultados en *name*, para los hubs entre *n* y *N*.
    - main(p): Realiza un entremiento sobre p hubs usando SA y GA y grafica los resultados.
    - make_all_table(): Crea las tabla que aparece en la memoria.
    - read_and_plot(): Crea los gráficos que aparecen en la memoria. 
