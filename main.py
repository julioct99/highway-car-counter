import numpy as np
import cv2
import pandas as pd

cap = cv2.VideoCapture('highway.avi')
frames_count, fps, width, height = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS), cap.get(
    cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = int(width)
height = int(height)
print(frames_count, fps, width, height)

# Crea un data frame de pandas con tantas filas como numero de fotogramas haya
df = pd.DataFrame(index=range(int(frames_count)))
df.index.name = "Frames"

# Posicion e incremento de las lineas (barreras)
horizontal1 = 225
horizontal2 = 250
incrementoLineas = 5

framenumber = 0  # Frame actual
carscrossedup = 0  # Total de coches que han pasado por el lado izquierdo
carscrosseddown = 0  # Total de coches que han pasado por el lado derecho
carids = []  # Lista que almacena los ids de los coches
caridscrossed = []  # Lista que almacena los ids de los coches que han cruzado las barreras
totalcars = 0  # Numero total de coches

fgbg = cv2.createBackgroundSubtractorMOG2()  # Background subtractor

ret, frame = cap.read()  # Se lee el frame
ratio = 0.75 # Ratio de reescalado
image = cv2.resize(frame, (0, 0), None, ratio, ratio)  # Reescalamos la imagen

while True:

    ret, frame = cap.read()

    if ret:  # Si hay frame, continuamos

        image = cv2.resize(frame, (0, 0), None, ratio, ratio)  # Reescala la imagen
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convierte la imagen a gris
        fgmask = fgbg.apply(gray)  # Usamos el background subtractor

        # Aplica diferentes umbrales a fgmask para intentar aislar los coches
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Kernel para aplicar a la morphology
        closing = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
        dilation = cv2.dilate(opening, kernel)
        retvalbin, bins = cv2.threshold(dilation, 220, 255, cv2.THRESH_BINARY)  # Borra las sombras

        # Crea los contornos
        contours, hierarchy = cv2.findContours(bins, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Se dibujan las lineas (barreras)
        cv2.line(image, (0, horizontal1), (width, horizontal1), (0, 0, 255), 2)
        cv2.line(image, (0, horizontal2), (width, horizontal2), (0, 255, 0), 2)

        # Area minima y maxima para los vehiculos
        minarea = 300
        maxarea = 50000

        # Listas para las coordenadas x e y de los centroides del frame actual
        cxx = np.zeros(len(contours))
        cyy = np.zeros(len(contours))

        for i in range(len(contours)):  # Itera los contornos del frame actual

            if hierarchy[0, i, 3] == -1:  # Usa la jerarquia para contar solo los contornos padre (que no esten dentro de otros)

                area = cv2.contourArea(contours[i])  # Area del contorno

                if minarea < area < maxarea:  # Comprueba si el area cae en nuestro rango

                    # Se calcula el centroide del contorno
                    cnt = contours[i]
                    M = cv2.moments(cnt)
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])

                    if cy > horizontal1:  # Se filtran los contornos que esten por encima de la linea

                        # Se obtienen los puntos del contorno (x, y, anchura, altura)
                        x, y, w, h = cv2.boundingRect(cnt)

                        # Crea un rectangulo alrededor del contorno
                        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

                        # Añade los centroides a la lista
                        cxx[i] = cx
                        cyy[i] = cy

        # Elimina los centroides que no han sido añadidos (valor 0)
        cxx = cxx[cxx != 0]
        cyy = cyy[cyy != 0]

        # Lista vacia para posteriormente comprobar los indices de los centroides añadidos al dataframe
        minx_index2 = []
        miny_index2 = []

        # Radio maximo de distancia entre dos centroides. Si estan mas cerca, se consideran el mismo
        maxrad = 25

        # La sección siguiente gestiona los centroides y los asigna a antiguos y nuevos ids de

        if len(cxx):  # Si hay centroides en el area especificada

            if not carids:  # Si la lista de ids de coches esta vacia

                for i in range(len(cxx)):  # Itera todos los centroides

                    carids.append(i)  # Añade el id de un coche a la lista
                    df[str(carids[i])] = ""  # Añade la columna correspondiente al coche al dataframe

                    # Asigna los valores del centroide con el frame actual (fila) y su id (column)
                    df.at[int(framenumber), str(carids[i])] = [cxx[i], cyy[i]]

                    totalcars = carids[i] + 1  # Incrementa el numero de coches

            else:  # Si ya hay ids de coches en la lista

                dx = np.zeros((len(cxx), len(carids)))  # Nuevos arrays para calcular deltas
                dy = np.zeros((len(cyy), len(carids)))  # Nuevos arrays para calcular deltas
                # Los deltas son diferencias de distancia entre un centroide antiguo y uno nuevo

                for i in range(len(cxx)):  # Itera los centroides

                    for j in range(len(carids)):  # Itera los ids de coches almacenados

                        # Obtiene el centroide del frame anterior para un id especifico de un coche
                        oldcxcy = df.iloc[int(framenumber - 1)][str(carids[j])]

                        # Obtiene el centroide del frame actual, que no tiene por que ser el calculado anteriormente
                        curcxcy = np.array([cxx[i], cyy[i]])

                        if not oldcxcy:  # Comprueba si el antiguo centroide esta vacio, por si el coche antiguo sale y entra uno nuevo
                            continue  # Continua al siguiente id de coche

                        else:  # Calcula los deltas del centroide para comparar la posicion actual posteriormente
                            dx[i, j] = oldcxcy[0] - curcxcy[0]
                            dy[i, j] = oldcxcy[1] - curcxcy[1]

                for j in range(len(carids)):  # Itera los ids de coches actuales

                    sum = np.abs(dx[:, j]) + np.abs(dy[:, j])  # Suma los wrt de los deltas a los ids de los coches

                    # Encuentra que indice de id de coche tiene la minima diferencia. Esto es el true index
                    correctindextrue = np.argmin(np.abs(sum))
                    minx_index = correctindextrue
                    miny_index = correctindextrue

                    # Obtiene los valores delta de los minimos deltas para comprobar si se esta dentro del radio mas adelante
                    mindx = dx[minx_index, j]
                    mindy = dy[miny_index, j]

                    if mindx == 0 and mindy == 0 and np.all(dx[:, j] == 0) and np.all(dy[:, j] == 0):
                        # Comprueba si el valor minimo es 0 y si todos los deltas son 0. Los deltas podrian ser cero
                        # si no se hubiera movido el centroide
                        continue 

                    else:
                        # Si los valores delta son menos que el radio maximo entonces se añade un centroide a ese id de coche 
                        # especifico
                        if np.abs(mindx) < maxrad and np.abs(mindy) < maxrad:

                            # Añade el centroide al id de coche existente anterior
                            df.at[int(framenumber), str(carids[j])] = [cxx[minx_index], cyy[miny_index]]
                            minx_index2.append(minx_index)  # Añade todos los indices que fueron añadidos a ids de coches previos
                            miny_index2.append(miny_index)

                for i in range(len(cxx)):  # Iteramos todos los centroides

                    # Si el centroide no esta en la lista minindex entonces se necesita añadir otro coche
                    if i not in minx_index2 and miny_index2:

                        df[str(totalcars)] = ""  # Crea otra columna con el numero total de coches
                        totalcars = totalcars + 1  # Añade otro coche al total
                        t = totalcars - 1  # t representa el numero de coches antes del incremento
                        carids.append(t)  # Lo añadimos a la lista de ids de coches
                        df.at[int(framenumber), str(t)] = [cxx[i], cyy[i]]  # Añadimos el centroide al nuevo id de coche

                    elif curcxcy[0] and not oldcxcy and not minx_index2 and not miny_index2:
                        # Comprueba si el centroide actual existe pero el previo no.
                        # Se añade un nuevo coche si minx_index2 esta vacia

                        df[str(totalcars)] = ""  # Crea otra columna con el numero total de coches
                        totalcars = totalcars + 1  # Añade otro coche al total
                        t = totalcars - 1  # t representa el numero de coches antes del incremento
                        carids.append(t)  # Lo añadimos a la lista de ids de coches
                        df.at[int(framenumber), str(t)] = [cxx[i], cyy[i]]  # Añadimos el centroide al nuevo id de coche

        currentcars = 0  # Numero de coches en pantalla
        currentcarsindex = []  # Indices de los ids de los coches que estan en pantalla actualmente
        
        for i in range(len(carids)):  # Itera todos los ids de coches

            if df.at[int(framenumber), str(carids[i])] != '':
                # Comprueba el frame actual para ver los ids de coches que estan activos, 
                # comprobando si el centroide existe en el frame para un determinado id
                currentcars = currentcars + 1  # Incrementa el numero de coches en pantalla
                currentcarsindex.append(i)  # Añade el id del coche a la lista de coches en pantalla

        for i in range(currentcars):  # Itera sobre los ids de coches en pantalla

            # Obtiene el centroide de un determinado id de coche para el frame actual
            curcent = df.iloc[int(framenumber)][str(carids[currentcarsindex[i]])]

            # Obtiene el centroide de un determinado id de coche para el frame anterior
            oldcent = df.iloc[int(framenumber - 1)][str(carids[currentcarsindex[i]])]

            if curcent:  # Si existe el centroide actual

                if oldcent:  # Comprueba si existe el centroide antiguo
                    # Comprobamos la posicion relativa con las barreras y si el centroide ya esta almacenado
                    if oldcent[1] >= horizontal2 and curcent[1] <= horizontal2 and carids[
                        currentcarsindex[i]] not in caridscrossed:

                        carscrossedup = carscrossedup + 1
                        cv2.line(image, (0, horizontal2), (width, horizontal2), (0, 0, 255), 5)
                        caridscrossed.append(
                            currentcarsindex[i])  # Añade el id del coche a la lista de coches contados para prevenir que se vuelva a contar

                    # Comprueba la posicion relativa del centroide antiguo respecto a las barreras
                    # para contar coches y ver los que aun no se han contado
                    elif oldcent[1] <= horizontal2 and curcent[1] >= horizontal2 and carids[
                        currentcarsindex[i]] not in caridscrossed:

                        carscrosseddown = carscrosseddown + 1
                        cv2.line(image, (0, horizontal2), (width, horizontal2), (0, 0, 125), 5)
                        caridscrossed.append(currentcarsindex[i])

        # Se muestra el texto informativo en la esquina superior izquierda
        cv2.rectangle(image, (0, 0), (335, 70), (0, 0, 0), -1)  # background
        cv2.putText(image, f"Vehiculos [izquierda]: {carscrossedup}", (0, 15), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 170, 0))
        cv2.putText(image, f"Vehiculos [derecha]: {carscrosseddown}", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 170, 0))
        cv2.putText(image, f"Frame {framenumber}/{int(frames_count)}", (0, 45), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 170, 0))
        cv2.putText(image, "Pulsa w/s para subir/bajar las barreras", (0, 60), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 170, 0))

        # Se muestran las imagenes
        cv2.imshow("real video", image)
        cv2.imshow("fgmask", fgmask)

        framenumber = framenumber + 1

        k = cv2.waitKey(int(1000/fps)) & 0xff  # int(1000/fps) es la velocidad normal, ya que waitkey esta en ms
        if k == 27:
            break
        elif (k == 119 or k == 87) and horizontal1 - incrementoLineas >= 0:  # Tecla w
            horizontal1 -= incrementoLineas
            horizontal2 -= incrementoLineas
        elif (k == 115 or k == 83) and horizontal2 + incrementoLineas <= (height * ratio - 30):  # Tecla s
            horizontal1 += incrementoLineas
            horizontal2 += incrementoLineas

    else:  # Si se ha terminado el video, terminamos el bucle
        break

cap.release()
cv2.destroyAllWindows()
