import cv2
import numpy as np
import pandas as pd

cap = cv2.VideoCapture("highway.avi")

frames_count, fps, width, height = (
    cap.get(cv2.CAP_PROP_FRAME_COUNT),
    cap.get(cv2.CAP_PROP_FPS),
    cap.get(cv2.CAP_PROP_FRAME_WIDTH),
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
)

width = int(width)
height = int(height)
print(frames_count, fps, width, height)

df = pd.DataFrame(index=range(int(frames_count)))
df.index.name = "Frames"

horizontal1 = 225
horizontal2 = 250
barIncrement = 5

carids = []
caridscrossed = []
totalcars = 0
framenumber = 0
carscrossedup = 0
carscrosseddown = 0


ratio = 0.75
ret, frame = cap.read()
fgbg = cv2.createBackgroundSubtractorMOG2()
image = cv2.resize(frame, (0, 0), None, ratio, ratio)

while True:
    ret, frame = cap.read()

    if ret:
        image = cv2.resize(frame, (0, 0), None, ratio, ratio)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fgmask = fgbg.apply(gray)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closing = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
        dilation = cv2.dilate(opening, kernel)
        retvalbin, bins = cv2.threshold(dilation, 220, 255, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(
            bins, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.line(image, (0, horizontal1), (width, horizontal1), (0, 0, 255), 2)
        cv2.line(image, (0, horizontal2), (width, horizontal2), (0, 255, 0), 2)

        minarea = 300
        maxarea = 50000

        cxx = np.zeros(len(contours))
        cyy = np.zeros(len(contours))

        for i in range(len(contours)):
            if hierarchy[0, i, 3] == -1:
                area = cv2.contourArea(contours[i])

                if minarea < area < maxarea:
                    cnt = contours[i]
                    M = cv2.moments(cnt)
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    if cy > horizontal1:
                        x, y, w, h = cv2.boundingRect(cnt)

                        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        cxx[i] = cx
                        cyy[i] = cy

        cxx = cxx[cxx != 0]
        cyy = cyy[cyy != 0]

        minx_index2 = []
        miny_index2 = []

        maxrad = 25

        if len(cxx):
            if not carids:
                for i in range(len(cxx)):
                    carids.append(i)
                    df[str(carids[i])] = ""

                    df.at[int(framenumber), str(carids[i])] = [cxx[i], cyy[i]]

                    totalcars = carids[i] + 1

            else:
                dx = np.zeros((len(cxx), len(carids)))
                dy = np.zeros((len(cyy), len(carids)))

                for i in range(len(cxx)):
                    for j in range(len(carids)):
                        oldcxcy = df.iloc[int(framenumber - 1)][str(carids[j])]
                        curcxcy = np.array([cxx[i], cyy[i]])

                        if not oldcxcy:
                            continue
                        else:
                            dx[i, j] = oldcxcy[0] - curcxcy[0]
                            dy[i, j] = oldcxcy[1] - curcxcy[1]

                for j in range(len(carids)):
                    sum = np.abs(dx[:, j]) + np.abs(dy[:, j])

                    correctindextrue = np.argmin(np.abs(sum))
                    minx_index = correctindextrue
                    miny_index = correctindextrue

                    mindx = dx[minx_index, j]
                    mindy = dy[miny_index, j]

                    if (
                        mindx == 0
                        and mindy == 0
                        and np.all(dx[:, j] == 0)
                        and np.all(dy[:, j] == 0)
                    ):
                        continue

                    else:
                        if np.abs(mindx) < maxrad and np.abs(mindy) < maxrad:
                            df.at[int(framenumber), str(carids[j])] = [
                                cxx[minx_index],
                                cyy[miny_index],
                            ]
                            minx_index2.append(
                                minx_index
                            )  # Añade todos los indices que fueron añadidos a ids de coches previos
                            miny_index2.append(miny_index)

                for i in range(len(cxx)):
                    if i not in minx_index2 and miny_index2:
                        df[str(totalcars)] = ""
                        totalcars = totalcars + 1
                        t = totalcars - 1
                        carids.append(t)
                        df.at[int(framenumber), str(t)] = [cxx[i], cyy[i]]

                    elif (
                        curcxcy[0]
                        and not oldcxcy
                        and not minx_index2
                        and not miny_index2
                    ):
                        df[str(totalcars)] = ""
                        totalcars = totalcars + 1
                        t = totalcars - 1
                        carids.append(t)
                        df.at[int(framenumber), str(t)] = [cxx[i], cyy[i]]

        currentcars = 0
        currentcarsindex = []

        for i in range(len(carids)):
            if df.at[int(framenumber), str(carids[i])] != "":
                currentcars = currentcars + 1
                currentcarsindex.append(i)

        for i in range(currentcars):
            curcent = df.iloc[int(framenumber)][str(carids[currentcarsindex[i]])]
            oldcent = df.iloc[int(framenumber - 1)][str(carids[currentcarsindex[i]])]

            if curcent:
                if oldcent:
                    if (
                        oldcent[1] >= horizontal2
                        and curcent[1] <= horizontal2
                        and carids[currentcarsindex[i]] not in caridscrossed
                    ):
                        carscrossedup = carscrossedup + 1
                        cv2.line(
                            image,
                            (0, horizontal2),
                            (width, horizontal2),
                            (0, 0, 255),
                            5,
                        )
                        caridscrossed.append(currentcarsindex[i])

                    elif (
                        oldcent[1] <= horizontal2
                        and curcent[1] >= horizontal2
                        and carids[currentcarsindex[i]] not in caridscrossed
                    ):
                        carscrosseddown = carscrosseddown + 1
                        cv2.line(
                            image,
                            (0, horizontal2),
                            (width, horizontal2),
                            (0, 0, 125),
                            5,
                        )
                        caridscrossed.append(currentcarsindex[i])

        cv2.rectangle(image, (0, 0), (335, 70), (0, 0, 0), -1)
        cv2.putText(
            image,
            f"Vehicles [left]: {carscrossedup}",
            (0, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 170, 0),
        )
        cv2.putText(
            image,
            f"Vehicles [right]: {carscrosseddown}",
            (0, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 170, 0),
        )
        cv2.putText(
            image,
            f"Frame {framenumber}/{int(frames_count)}",
            (0, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 170, 0),
        )
        cv2.putText(
            image,
            "Press w/s to rise/lower the barriers",
            (0, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 170, 0),
        )

        cv2.imshow("real video", image)
        cv2.imshow("fgmask", fgmask)

        framenumber = framenumber + 1

        k = cv2.waitKey(int(1000 / fps)) & 0xFF
        if k == 27:
            break
        elif (k == 119 or k == 87) and horizontal1 - barIncrement >= 0:  # w
            horizontal1 -= barIncrement
            horizontal2 -= barIncrement
        elif (k == 115 or k == 83) and horizontal2 + barIncrement <= (
            height * ratio - 30
        ):  # s
            horizontal1 += barIncrement
            horizontal2 += barIncrement

    else:
        break

cap.release()
cv2.destroyAllWindows()
