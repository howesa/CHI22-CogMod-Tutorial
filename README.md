# CogMod-Tutorial
CogMod-Tutorial is a set of Python Jupyter Notebooks designed as an introduction to cognitive, neural and RL modeling in Human-Computer Interaction. 

Designed for delivery at CHI'2022 by Jussi Jokinen, Antti Oulasvirta and Andrew Howes with thanks to Xiuli Chen.

### Getting started

1. In your browser navigate to Google Colab and create a new notebook (link in bottom right of the pop-up).

2. In a code cell, mount Google drive

```
  from google.colab import drive
  drive.mount('/content/drive')
```

3. Generate a personal access token for github

For this you will need your personal access token. Go to your github web page and click on your photo in the top right corner. Next, click on "settings" and then "developer settings". Then click on  Personal access tokens and "Generate token".

Copy your personal access token to the clipboard and keep a copy of it somewhere safe.

Do not share your personal access token.

4. Clone the CHI22-CogMod-Tutorial github repository

Enter the following command with your personal access token inserted:

!git clone https://{personal access token}github.com/howesa/CHI22-CogMod-Tutorial

5. Change directory into the git folder

%cd CHI22-CogMod-Tutorial

6. configure your email and name

!git config --global user.email "howesa@mac.com"

!git config --global user.name "Andrew Howes"

8. Modify the repository (developers only!) using !git add, !git commit, !git push

9. Congratulations.
