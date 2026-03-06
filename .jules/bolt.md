## 2024-05-24 - [Minimize memory spikes when loading large directories]
**Learning:** `os.scandir` combined with generator expressions greatly minimizes memory spikes when parsing directory listings containing many files compared to `os.listdir` with list comprehensions.
**Action:** Always prefer `os.scandir` and generator expressions for loading/iterating through files in potentially large directories, avoiding materializing the entire list of filenames in memory.
