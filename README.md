# Лабораторные работы по математической статистике ПМИ СПБПУ весна 2026

## [Отчёт](report/)

## Использование:
0. `python -m venv venv` 
1. Linux - `source venv/bin/activate`
2. Windows - `venv\Scripts\activate` 
3. `pip install -r requirements.txt`
4. `python lab1_histograms.py`
8. Результаты в папке [`results`](results/)
9. Отчеты в папке [`report`](report/)


## Как собрать отчёт?
 - вам понадобиться [`typst`](https://typst.app/open-source/#download)
 - `cd report/lab1 && typst compile --root ../../ ./lab1.typ && cd ../../` - для первой лабы
 - Отчет появится в папке с `.typ`