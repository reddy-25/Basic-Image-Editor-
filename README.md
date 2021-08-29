# Basic-Image-Editor
This is a simple GUI based image editor which can perform certain image processing techniques and certain functionalities.  
Actually the file code is run on pycharm which has the run button.
The version of python used is 3.9.5.  
The libraries required are given in the requirements.txt  
The main code is written in main.py  
If you unzip the folder you will find the main.py in the folder itself. Now you have to notedown the path to main.py or open the terminal in the directory where main.py is present. 
### Shell command
If you open the terminal/shell in the directory where main.py is present, the command is  
python3 main.py  
Otherwise the command is   
python3 pathname  
where pathname is the path to the file main.py  
I have run the main.py file on pycharm. So I am not exactly sure of the command. But I think the above one works.
### Description
Once the required modules are installed and the command is run a tkinter GUI will appear with the following image proxessing techniques and functionalities.  
**Image processing techniques:**  
Global Histogram equalisation  
Local Histogram equalisation  
Logarithmic Transformation  
Sharpening  
Blurring  
Negative of an image  
Gamma transformation
  
Image processing techniques like Gamma transformation, Sharpening and Blurring require entries.  
  
Gamma transformation requires the user to enter **gamma** value which is greater than 0.  
If gamma>1, then we can see the information in the brighter images clearly.  
If  0<gamma<1, then we can see the information in the darker images clearly.  
  
Blurring requires the user to enter the **filter size**. Filter size has to be odd positive integer. Filter implemented for blurring is box filter.  
If the filter size is increased, the level of blurring also increases.  
  
Sharpening requires the user to enter the **sigma value** greater than 0. This sigma is used to calculate the filter coefficients in gaussian filter.  
As sigma increases gaussian blurring increases and this increase the sharpening. 
Once the gaussian filter is formed we will perform correlation which smoothens the image. Now the mask is formed by subtracting   
the smoothened image from the original image. Now certain percent of mask is added to the original signal which sharpens the image.  
  
The reamining image processing techniques are straightforward and are just buttons and if you click them the resultant images after transformation will be displayed.  

**Functionalities:**  
Loading the image- This button asks the user to load the image.  
Saving the image- This button asks the user to select the path where to save.  
Dispalying image area- This button displays the image area of the laoded image  
Exit-This button closed the tkinter GUI.  
Undo - This button can take you back upto two images.  
Undo all- This button brings back the original image.
