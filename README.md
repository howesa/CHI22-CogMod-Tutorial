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

Run the code cell to mount the drive.

3. Clone the CHI22-CogMod-Tutorial github repository

Enter:
```
!git clone https://github.com/howesa/CHI22-CogMod-Tutorial /content/drive/My\ Drive/Colab\ Notebooks/CHI22_CogMod_Tutorial```
```
When successful you should see a last line which is something like:

Resolving deltas: 100% (...) done:

4. Go to the cloned folder

Navigate to your Google drive folder and click on the folder "Colab Notebooks"

Next click on "CHI22-CogMod_Tutorial"

You should see a set of folders including:

.git
00-preparatiion
01-Introduction

etc.

Well done! You have successfully cloned the github repository.  Now to get started open the file getting_started.ipynb and follow the instructions.
