import numpy as np
import cv2 as cv
from typing import List


class ImagePoint:

    def __init__(self, x:int, y:int):
        if type(x) != int or type(y) != int:
            raise ValueError("Point coordinates should be integers")
        self.x = x
        self.y = y

    def get_point_as_array(self) -> List[int]:
        return [self.x, self.y]

    def distance(self, another_point: "ImagePoint") -> int:
        return int(np.sqrt((self.x-another_point.x)**2+(self.y-another_point.y)**2))


class ImageBGR:

    def __init__(self, image: np.ndarray):
        if type(image) != np.ndarray:
            raise ValueError("ImageBGR can be constructed only from np.ndarray")

        self.__image: np.ndarray = image

    @classmethod
    def from_file(cls, filename: str) -> 'ImageBGR':
        """
        Create new ImageBGR from file represented by filename
        :param filename: path to file from which we want to create ImageBGR
        :return: ImageBGR
        """
        if type(filename) != str:
            raise ValueError("Only string as path is accepted")
        return cls(cv.imread(filename))

    @classmethod
    def from_array(cls, image: np.ndarray) -> 'ImageBGR':
        """
        Create new ImageBGR from array representing image
        :param image: ndarray representing image
        :return: ImageBGR
        """
        if type(image) != np.ndarray:
            raise ValueError("Only np.ndarray representing image is accepted")
        return cls(image)

    def gray(self) -> np.ndarray:
        """
        Funkce která vrací obraz ve stupních šedi
        """

        return cv.cvtColor(self.__image, cv.COLOR_BGR2GRAY)

    def lab(self) -> np.ndarray:
        """
        Funkce která vrací obraz v barevném prostoru Lab
        """

        return cv.cvtColor(self.__image, cv.COLOR_BGR2LAB)

    def rgb(self) -> np.ndarray:
        """
        Funkce která vrací obraz v RGB
        """
        return cv.cvtColor(self.__image, cv.COLOR_BGR2RGB)

    def bgr(self) -> np.ndarray:
        """
        Funkce která vrací obraz v BGR
        """
        return self.__image

    def resize(self, width: int, height: int) -> 'ImageBGR':
        """
        Funkce která vrací novou instanci ImageBGR obsahující obraz z původní instance třídy ImageBGR ale s novými rozměry width a height.
        """
        if type(width) != int or type(height) != int or width < 0 or height < 0:
            raise ValueError("Width and height must be integers greater than 0.")

        return ImageBGR.from_array(cv.resize(self.__image, (width, height)))

    def rotate(self, angle: int, keep_ratio: bool) -> 'ImageBGR':
        """
        Funkce která vrací novou instanci ImageBGR obsahující obraz z původní instance třídy ImageBGR ale s novými rozměry width a height.
        Pokud je nastaveno keep_ratio na True, nový obraz musí mít stejný rozměr jako původní. Pokud je nastaveno na False, nový obraz musí
        obsahovat celou obrazovou informaci z původního obrazu.
        """
        (height, width) = self.shape[:2]
        (cX, cY) = (width // 2, height // 2)  # coordinates of center
        rotation_matrix = cv.getRotationMatrix2D((cX, cY), angle, 1.0)

        if keep_ratio:
            return ImageBGR.from_array(cv.warpAffine(self.__image, rotation_matrix, (width, height)))
        else:
            cos = np.abs(rotation_matrix[0, 0])
            sin = np.abs(rotation_matrix[0, 1])

            new_width = int((height * sin) + (width * cos))
            new_height = int((height * cos) + (width * sin))

            rotation_matrix[0, 2] += (new_width / 2) - cX
            rotation_matrix[1, 2] += (new_height / 2) - cY

            return ImageBGR.from_array(cv.warpAffine(self.__image, rotation_matrix, (new_width, new_height)))

    def histogram(self) -> np.ndarray:
        """
        Funkce vrací histogram obrazu z jeho verze ve stupních šedi.
        """

        return cv.calcHist(images=[self.gray()], channels=[0], mask=None, histSize=[256], ranges=[0, 256])

    def save_as_bmp(self, path: str, name: str):
        """
        Uloží obrázek jako bmp
        :param path: cesta kam chci soubor uložit
        :param name: jméno souboru
        """
        if type(path) != str or type(name) != str:
            raise ValueError("Path and name should be strings.")
        if path[-1] == "/":
            path = path[:(len(path)-1)]

        cv.imwrite(path+"/"+name+".bmp", self.__image)

    def perspective_transform(self, points: List[ImagePoint]) -> "ImageBGR":
        """
        Pomocí ručně zadaných rohů zarovná obrázek
        """
        if len(points) != 4:
            raise ValueError("Number of points should be 4")

        final_image_width = points[0].distance(points[1])
        final_image_height = points[0].distance(points[2])
        final_points = [ImagePoint(0, 0),
                        ImagePoint(final_image_width, 0),
                        ImagePoint(0, final_image_height),
                        ImagePoint(final_image_width, final_image_height)]

        matrix = cv.getPerspectiveTransform(np.float32([p.get_point_as_array() for p in points]),
                                            np.float32([p.get_point_as_array() for p in final_points]))

        return ImageBGR.from_array(cv.warpPerspective(self.__image, matrix, (final_image_width, final_image_height)))

    def automatic_perspective_transform(self) -> "ImageBGR":
        """
        Automaticky detekuje rohy a ořízne vyrovná obrázek
        """
        return self.perspective_transform(self.corners)

    @property
    def shape(self) -> tuple:
        """
        Funkce dekorovaná jako atribut která vrací rozměry uloženého obrazu.
        """

        return self.__image.shape


    @property
    def size(self) -> int:
        """
        Funkce dekorovaná jako atribut která vrací obrazem obsazenou paměť (čistě polem do kterého je obraz uložen).
        """

        return self.__image.size

    @property
    def width(self) -> int:
        """
        Funkce dekorovaná jako atribut která vrací šířku obrazu.
        """

        return self.__image.shape[1]

    @property
    def height(self) -> int:
        """
        Funkce dekorovaná jako atribut která vrací výšku obrazu.
        """

        return self.__image.shape[0]

    @property
    def corners(self) -> List[ImagePoint]:
        """
        Funkce dekorovaná jako atribut která vrací rohy rozpoznaného obrázku.
        """

        preprocessed_image = self.gray()
        preprocessed_image = cv.GaussianBlur(preprocessed_image, (5, 5), 1)  # KERNAL -> must be odd, SIGMAX
        preprocessed_image = cv.Canny(preprocessed_image, 200, 200)  # edge detector
        kernel = np.ones((2, 2))
        preprocessed_image = cv.dilate(preprocessed_image, kernel, iterations=2)
        preprocessed_image = cv.erode(preprocessed_image, kernel, iterations=1)

        contours, hierarchy = cv.findContours(preprocessed_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        biggest: np.ndarray = np.array([])
        maxArea = 0
        for c in contours:
            area = cv.contourArea(c)
            if area > (self.width*self.height)*0.1:
                peri = cv.arcLength(c, True)  # length of perimeter
                approx = cv.approxPolyDP(c, 0.02 * peri, True)
                if area > maxArea and len(approx) == 4:
                    biggest = approx
                    maxArea = area

        return [ImagePoint(a.item(0), a.item(1)) for a in self.__reorder_points(biggest)]

    @staticmethod
    def __reorder_points(points: np.ndarray) -> np.ndarray:
        points = points.reshape((4, 2))
        reordered_points = np.zeros((4, 1, 2), np.int32)
        add = points.sum(1)

        reordered_points[0]  = points[np.argmin(add)]
        reordered_points[3] = points[np.argmax(add)]

        diff = np.diff(points, axis=1)

        reordered_points[1] = points[np.argmin(diff)]
        reordered_points[2] = points[np.argmax(diff)]

        return reordered_points







