# CogMod-Tutorial

CogMod-Tutorial is a set of Python Jupyter Notebooks designed as an introduction to cognitive, neural and RL modeling in Human-Computer Interaction. 

Designed for delivery at CHI'2022 by Jussi Jokinen, Antti Oulasvirta and Andrew Howes with thanks to Xiuli Chen.

### Getting started

1. In a Chrome browser navigate to Google Colab and create a new notebook (link in bottom right of the pop-up).

2. In a code cell, mount Google drive

```
  from google.colab import drive
  drive.mount('/content/drive')
```

Run the code cell and mount Google drive.

3. Clone the CHI22-CogMod-Tutorial github repository

Enter the following command with your personal access token inserted:

```
!git clone https://github.com/howesa/CHI22-CogMod-Tutorial
```

When successful you should see something like:
```
Resolving deltas... done:...
```

4. Change directory into the git folder

```
%cd 'CHI22-CogMod-Tutorial'
```
