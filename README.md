# CogMod-Tutorial

CogMod-Tutorial is a set of Python Jupyter Notebooks designed as an introduction to cognitive, neural and RL modeling in Human-Computer Interaction. 

Designed for delivery at CHI 2022 by Jussi Jokinen, Antti Oulasvirta and Andrew Howes with thanks to Xiuli Chen.

### Getting started

1.  In a Chrome browser navigate to your Google Drive.
2.  In the top left corner click on the "+ New" button.
3.  Create a new folder called "CHI22" -- the exact name is important.
4.  Navigate into the folder
5.  In the top left corner click on the "+ New" button again.
6.  This time click on "More" at the bottom of the pop-up.
7.  Click on "Google Colaboratory" to create a new notebook in Colab.
9.  Paste the following lines into the code cell (a long text window) in the notebook:

```
  from google.colab import drive
  drive.mount('/content/drive')
  !git clone https://github.com/howesa/CHI22-CogMod-Tutorial /content/drive/My\ Drive/CHI22/CHI22_CogMod_Tutorial
```

Run the code cell by pressing the triangle in a circle button to the left of the code cell.

You will be asked whether you want to permit this notebook to access your google drive. Say, yes.

If this is successful then you should see something like the following (though numbers will differ):

```
Mounted at /content/drive
Cloning into '/content/drive/My Drive/CHI22/CHI22_CogMod_Tutorial'...
remote: Enumerating objects: 149, done.
remote: Counting objects: 100% (149/149), done.
remote: Compressing objects: 100% (130/130), done.
remote: Total 149 (delta 72), reused 47 (delta 14), pack-reused 0
Receiving objects: 100% (149/149), 2.09 MiB | 7.52 MiB/s, done.
Resolving deltas: 100% (72/72), done.
```

10. Navigate to the new folder that you have just created. One way to do this is to create a new tab in your browser and then going to drive. The new folder should be on the path: My Drive > CHI22 > CHI22_CogMod_Tutorial. Otherwise, f you go to the old drive tab then you must refresh it to see the new folder.

You should see a set of files and folders that include the file getting_started.ipynb. Open this file and follow the instructions.
