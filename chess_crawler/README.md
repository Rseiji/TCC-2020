# How to use this crawler

* First, you must execute `run_all.py` in `Python 2`, folder `ChessComGen_Python2`, and copy folder `saved_files` to this project's root.

<br>

* To get the raw data, from `TCC-2020/chess_crawler` folder, you must:

```
cd chess_crawler && scrapy crawl chess_spider -o <raw data file name>.csv
```

<br>

* Now, return to `TCC-2020/chess_crawler` and run:

```
python3 csv_parse.py ./chess_crawler/<file generated from scrapy>.csv <formated file>.csv
```

<br>

* The `.csv` output of `csv_parse.py` contains each comment associated with html page identification and associated moves.
