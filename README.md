# Klasyfikacja materiałów wideo

## Wprowadzenie
Projekt dotyczy klasyfikacji materiałów wideo (action recognition) z podzbioru danych [UCF101](https://www.crcv.ucf.edu/data/UCF101.php). Dla uproszczenia problemu ze 101 klas w wyjściowym datasecie zostało wybranych 8 pierwszych klas. Są to kolejno:
- ApplyEyeMakeup
- ApplyLipstick
- Archery
- BabyCrawling
- BalanceBeam
- BandMarching
- BaseballPitch
- BasketballDunk

Model wykorzystuje konwolucje 3D i osiąga accuracy około 70% na zbiorze testowym. Model jest zaimplementowany przy użyciu Tensorflow.

## Uruchomienie
Wystarczy sklonować repozytorium, zainstalować biblioteki z requirements.txt (istotne jest wykorzystanie tensorflow 2.15.0, ponieważ dla wersji 2.16.1 pojawiają się błędy w warstwie Flatten()), a następnie uruchomić main.py. Z wykorzystaniem GPU model powinien trenować się w ciągu kilkunastu minut.

## Funkcjonalności
W celu zachowania klarowności struktury kodu zaimplementowano następujące klasy:

- DataLoader - pozwala na wygodne wczytanie danych i podział na train, val, test. Możliwy jest wybór liczby wczytanych klas z UCF101, liczby klatek reprezentujących materiał wideo, rozmiaru batch'ów, a także liczby obserwacji w zbiorach train, val, test.
- ModelTrainer - definiuje architekturę modelu, pozwala na wybór parametrów learning_rate oraz epochs i wytrenowuje model na zadanych zbiorach train, val.
- ModelEvaluator - rysuje wykresy krzywych uczenia dla modelu wytrenowanego przez ModelTrainer. Dla zbioru testowego zwraca wartości accuracy i loss, następnie precision i recall dla każdej z klas, a także wizualizuje confusion matrix.

## Dataset
Khurram Soomro, Amir Roshan Zamir and Mubarak Shah, [UCF101: A Dataset of 101 Human Action Classes From Videos in The Wild.](https://www.crcv.ucf.edu/papers/UCF101_CRCV-TR-12-01.pdf), CRCV-TR-12-01, November, 2012.

## Źródła
Fragmenty kodu dotyczące ładowania danych, a także ewaluacji modelu bazują na tutorialu [Video classification with a 3D convolutional neural network](https://www.tensorflow.org/tutorials/video/video_classification). 

## Autorzy
- Aleksandra Talabska
- Szymon Jasiński