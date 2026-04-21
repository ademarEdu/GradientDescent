import numpy as np
def cargar_lena_pgm(ruta):
    with open(ruta, 'r') as f:
        # 1. Leer todas las palabras del archivo ignorando líneas que empiezan con '#'
        # Esto es necesario porque el formato P2 puede tener saltos de línea arbitrarios
        datos = []
        for linea in f:
            if not linea.startswith('#'):
                datos.extend(linea.split())

    # 2. Extraer el encabezado (Header)
    formato = datos[0]       # Debería ser 'P2'
    ancho = int(datos[1])    # 512
    alto = int(datos[2])     # 512
    max_gris = int(datos[3]) # 245

    # 3. Convertir el resto de los datos a números enteros
    # Estos son los valores de los píxeles (del 4 en adelante)
    pixeles = np.array(datos[4:], dtype=np.float64)

    # 4. Re-formar el vector en una matriz de 512x512
    # Aquí es donde el vector 1D se vuelve la imagen 2D
    imagen = pixeles.reshape((alto, ancho))

    return imagen, ancho, alto, max_gris, formato

# Uso del código
# imagen_matriz, w, h, m, f = cargar_lena_pgm(ruta)
# print(f"Formato de mi imagen es {f}")
# print(f"Pixel de mayor tamaño {m}")
# print(f"Imagen cargada con éxito: {w}x{h} píxeles.")
# print(f"El valor del primer píxel es: {imagen_matriz[0, 0]}")
#Convirtiendo mi imgane a un vector (Matriz a vector), el vector es de tamaño w*h
# def Flattening(imagen_matriz):
#     h, w = imagen_matriz.shape #Obtener el ancho y alto de mi matriz
#     vector_de_matriz = np.zeros(w * h)#Determinar el tamaño de mi vector 
#     for i in range (h):
#         for j in range (w):
#             index = i * w + j # Formula para indexar mi matriz 
#             vector_de_matriz[index] = imagen_matriz[i, j]

#     return vector_de_matriz

# print("EL tamaño del vector es:", Flattening(imagen_matriz))
#COnvirtiendo mi matriz en varios vectores para mejor lectura 
#Vectores son los valores por parte de las filas
def Flattening_parts (imagen_matriz):
    h, w = imagen_matriz.shape
    vectores = []
    for i in range (h):
        vector = np.zeros(w)
        for j in range(w):    
            vector[j] = imagen_matriz[i,j]
        vectores.append(vector)
    return vectores
# mis_vectores = Flattening_parts(imagen_matriz)
# print("Vectores ", mis_vectores[0][:15])
#Pasar de vector a magtriz 
def vector_to_matriz (mis_vectores, w, h):
    matriz = np.zeros((h , w))
    vector_largo = np.concatenate(mis_vectores)
    for k in range(len(vector_largo)):
        i = k // w
        j = k % w
        matriz[i, j] = vector_largo[k]
    
    return matriz
# mi_matriz = vector_to_matriz(mis_vectores, w, h)
#FUncion para guardar la imagen en pgm
def Guardar_como_PGM(nombre_archivo, matriz, max_gris):
    h, w = matriz.shape
    with open(nombre_archivo, 'w') as f:
        # 1. El "Número Mágico"
        f.write("P2\n")
        # 2. Las dimensiones
        f.write(f"{w} {h}\n")
        # 3. El valor máximo permitido
        f.write(f"{int(max_gris)}\n")
        
        datos_planos = matriz.flatten().astype(int)
        
    
        for i in range(0, len(datos_planos), 12):
            linea = " ".join(map(str, datos_planos[i : i + 12]))
            f.write(linea + "\n")
# Guardar_como_PGM("resultado_final.pgm", mi_matriz, 245)
#Funcion para guardar una imagen en png 
import matplotlib.pyplot as plt
def Guardar_como_imagen(matriz, nombre_archivo):
    plt.imsave(nombre_archivo, matriz, cmap='gray', vmin = 0, vmax = 245)
    print (f"Imagen guardada como {nombre_archivo}")
# Guardar_como_imagen(mi_matriz, "Lenna_suavizada.png")