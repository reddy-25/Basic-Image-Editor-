# modules required for the task
from tkinter import filedialog, messagebox  # to load, save and to show warnings.
from PIL import ImageTk  # to convert from mat(which we get throug PIL.Image) to image
import PIL.Image as Img  # in order to display on to the gui etc.
import numpy as np  # useful for array operations
import cv2  # in order to load image, convert to hsv etc.
import math  # for doing mathematical operations like log, exp
from tkinter import *  # tkinter is a standard GUI package


# class of Image editor
class Image_editor:  # main class for Image editor
    def __init__(self, root):  # init is like a constructor, root is the main frame
        self.root = root  # assigning main frame to variable root in the class
        self.image_stack = []  # to maintain a buffer for undoing
        self.stack_undo_index = None  # pointer to the current image in buffer
        self.max_size = 3  # maximum size of buffer
        self.original = None  # to store original image in the memory for undo all
        self.is_original = False  # whether original image is loaded or not on to the gui

        self.header = Label(self.root, text="Basic Image Editor", bg="#0019fc", fg="white", font=("arial", 20, "bold")) # main header in gui
        self.header.pack(fill=X) # this means the main header only expands horizontally on the gui

        self.blurr = Button(root, text="Blur:", font=("ariel 13 bold"), width=9, anchor='e', command=self.get_filter_size)  # creating blurring button on gui
        self.blurr.place(x=15, y=50)  # placing the button at the designated x, y position
        self.blurr["state"] = DISABLED  # when the image is not loaded onto the gui this button will be disabled
        self.blur_entry = Entry(self.root, width=15)  # this is used to get the filter size for blur
        self.blur_entry.place(x=120, y=55)  # placing the button on the gui at x,y position
        self.blur_entry["state"] = DISABLED  # when the image is not loaded onto the gui this button will be disabled

        self.sharp = Button(root, text="Sharpening:",font=("ariel 13 bold"), width=9, anchor='e', command=self.get_sigma)  # creating button for sharpening image
        self.sharp.place(x=8, y=100)  # palcing the button at the designated x,y position
        self.sharp["state"] = DISABLED # when the image is not loaded onto the gui this button will be disabled
        self.sharp_entry = Entry(root, width=15)  # this is used to get the variance value of gaussian filter for unsharp mssking
        self.sharp_entry.place(x=120, y=105)  # placing at x,y position
        self.sharp_entry["state"] = DISABLED  # when the image is not loaded onto the gui this button will be disabled


        self.local_hist = Button(root, text="local histogram equalisation", font=("ariel 13 bold"), command=self.local_histogram)  # creating button for local histogram
        self.local_hist.place(x=240, y=50)  # placing the button on gui
        self.local_hist["state"] = DISABLED  # when the image is not loaded onto the gui this button will be disabled

        self.global_hist = Button(root, text="global histogram equalisation", font=("ariel 13 bold"),
                                  command = self.global_hist)  # creating global histogram button for gui
        self.global_hist.place(x=240, y=100)  # placing button on gui
        self.global_hist["state"] = DISABLED  # # creating global histogram button for gui

        self.gamma = Button(root, text="gamma correction:", font=("ariel 13 bold"), command=self.get_gamma)  # creating button for gamma transformation
        self.gamma.place(x=500, y=50)  # placing it on gui
        self.gamma_entry = Entry(root,  width=10)  # entry for getting gamma value
        self.gamma_entry.place(x=670, y=60)  # placing the entry on gui
        self.gamma_entry["state"] = DISABLED  # when the image is not loaded onto the gui this button will be disabled
        self.gamma["state"] = DISABLED  # when the image is not loaded onto the gui this button will be disabled

        self.log = Button(root, text="logarithmic transformation", font=("ariel 13 bold"), command=self.log_transform)  # creating button for log transform
        self.log.place(x=500, y=100)  # palcing it onto gui
        self.log["state"] = DISABLED  # when the image is not loaded onto the gui this button will be disabled

        self.negate = Button(root, text="Negative of image",  font=("ariel 13 bold"),
                      command=self.negative)  # creating button for negating an image
        self.negate.place(x=280, y=600)  # placing it on gui
        self.negate["state"] = DISABLED  # when the image is not loaded onto the gui this button will be disabled

        self.undo = Button(root, text="Undo", bg='black', fg='gold', font=('ariel 15 bold'), relief=GROOVE,
                      command=self.undo)  # creaating button for undo
        self.undo.place(x=100, y=600)  # placing it on gui
        self.undo["state"] = DISABLED  # when the image is not loaded onto the gui this button will be disabled

        self.undo_all = Button(root, text="Undo all", bg='black', fg='gold', font=('ariel 15 bold'), relief=GROOVE,
                      command=self.undo_all)  # creating button for undo all
        self.undo_all.place(x=500, y=600)  # palcing it on gui
        self.undo_all["state"] = DISABLED  # when the image is not loaded onto the gui this button will be disabled

        self.display_area = Button(root, text="Display area", bg='black', fg='gold', font=('ariel 15 bold'), relief=GROOVE, command=self.display_image_area)  # creating button for dispalying image area
        self.display_area.place(x=650, y=610)  # placing it onto the gui
        self.display_area["state"] = DISABLED  # when the image is not loaded onto the gui this button will be disabled

        self.btn1 = Button(root, text="Load Image", bg='black', fg='gold', font=('ariel 15 bold'), relief=GROOVE,
                      command=self.open_image)  # creating button for loading an image
        self.btn1.place(x=100, y=650)  # placing it on gui

        self.save = Button(root, text="Save Image", width=12, bg='black', fg='gold', font=('ariel 15 bold'), relief=GROOVE,
                      command=self.savefile)  # creating button for saving image
        self.save.place(x=280, y=650)  # placing it onto gui
        self.save["state"] = DISABLED  # when the image is not loaded onto the gui this button will be disabled

        self.quit = Button(root, text="Exit", width=12, bg='black', fg='gold', font=('ariel 15 bold'), relief=GROOVE,
                      command=root.destroy)  # creating button for exit
        self.quit.place(x=460, y=650)  # placing it onto gui

        # create canvas to display image
        self.canvas2 = Canvas(root, width="760", height="400", bg="black", relief=RIDGE, bd=2)  # creating space for displaying image area
        self.canvas2.place(x=15, y=150)  # placing it onto gui

    def display_image_area(self):  # used to dispaly image area
        # this show the info about image area in pixels
        messagebox.showinfo(title="Image area in (width*height) pixels", message=str(self.y1_size)+"pixels x"+str(self.x1_size)+"pixels")

    def open_image(self):  # this is just for opening image
        self.filename = filedialog.askopenfilename(title="Select an image file")  # asks us to open file and store the file path in filename
        # filetypes=(("jpg files", "*.jpg"), ("png files", "*.png"), ("jpeg files", "*.jpeg")))

        self.my_img = cv2.imread(self.filename)  # reading the image with opencv
        if self.my_img is None:  # if no image is selected or wrong extension is selcted
            messagebox.showerror("Error", "File format not supported")  # shows error that file is not selected
        else:  # otherwise if valid file is selected
            self.bw = self.is_gray(self.my_img)  # determines whether image is color or black and white
            self.x1_size = self.my_img.shape[0]  # storing the hwight of image in pixels
            self.y1_size = self.my_img.shape[1]  # storing the width of image in pixesl
            dim = (600, 600)  # dimensions to resize the image since if we take large image then processing becomes hectic
            if self.x1_size>600 and self.y1_size>600:
                self.my_img = cv2.resize(self.my_img, dim, interpolation=cv2.INTER_AREA)  # resized image
                self.x_size = self.my_img.shape[0]  # height of resized image
                self.y_size = self.my_img.shape[1]  # width of resized image
            else:
                self.x_size = self.x1_size
                self.y_size = self.y1_size
            blue, green, red = cv2.split(self.my_img)  # cv2 will read image in bgr so getting independent channels
            self.image = np.array(cv2.merge((red, green, blue)))  # we will operate on arrays

            self.original = self.image.copy()  # saving original copy in the variable for undo_all
            self.is_original = True  # if original image is loaded or not is stored
            self.state_enable()  # enabling the states after image is loaded
            self.undo["state"] = DISABLED  # only disabling the undo since this is the first image
            self.image_stack = []  # this will store the images for undoing
            self.image_stack.append(self.original)  # storing the original image firstly in the stack which is current image

            self.stack_undo_index = 0  # this will act like a pointer which points to current image
            self.display_image(self.image)  # this will display the present image

    def display_image(self, disp_img):  # this function displays the image on the canvas of gui

        disp_img = cv2.resize(disp_img, (self.y1_size, self.x1_size), interpolation=cv2.INTER_AREA)  # first resizing to the original image
        self.rgb = Img.fromarray(disp_img)  # converting from array to mat format so that PIL can use to dispaly image
        self.rgb.thumbnail((400, 400))  # creating a thumbnail for dispaying onto the canvas of gui
        self.disp_img = ImageTk.PhotoImage(image=self.rgb)  # Image tk reads the mat format and will be ready to display

        self.canvas2.create_image(400, 205, image=self.disp_img)  # creating image to display from Imagetk
        self.canvas2.image = self.disp_img  # displaying on the canvas

    def state_disable(self):  # this function is used to diable the buttons while some function is being performed so that it doesn't interrupt
        self.display_area["state"] = DISABLED  # disabling display area
        self.local_hist["state"] = DISABLED  # disabling local histogram button
        self.global_hist["state"] = DISABLED  # disabling global histogram button
        self.gamma["state"] = DISABLED  # disabling gamma transformation button
        self.log["state"] = DISABLED  # disabling log transform button
        self.blurr["state"] = DISABLED  # disabling blur button
        self.blur_entry["state"] = DISABLED  # disabling blur entry button
        self.sharp_entry["state"] = DISABLED # disabling sharpening entry  button
        self.sharp["state"] = DISABLED  # disabling sharpening  button
        self.negate["state"] = DISABLED  # disabling neagative of image button
        self.undo["state"] = DISABLED  # disabling undo button
        self.undo_all["state"] = DISABLED  # disabling undo_all button
        self.save["state"] = DISABLED  # disabling save button
        self.gamma_entry["state"] = DISABLED  # disabling gamma transformation entry button
    def state_enable(self):  # enabling all button after specific function is performed
        self.display_area["state"] = NORMAL  # enabling display area button
        self.local_hist["state"] = NORMAL  # enabling local histogram button
        self.global_hist["state"] = NORMAL  # enabling global histogram button
        self.gamma["state"] = NORMAL  # enabling gamma transformation button
        self.log["state"] = NORMAL  # enabling log transformation button
        self.sharp["state"] = NORMAL  # enabling sharpenin button
        self.sharp_entry["state"] = NORMAL  # enabling sharpening entry button
        self.blurr["state"] = NORMAL  # enabling blurring  button
        self.blur_entry["state"] = NORMAL  # enabling blurring emtry button
        self.negate["state"] = NORMAL   # enabling negative of image button
        self.undo["state"] = NORMAL  # enabling undo button
        self.undo_all["state"] = NORMAL  # enabling undo_all button
        self.save["state"] = NORMAL  # enabling save button
        self.gamma_entry["state"] = NORMAL  # enabling gamma transformation entry label


    def stack_operation(self, cur_img):  # to perform stack operations
        size = len(self.image_stack)  # size of stack which stores images
        if(size == self.max_size):  # if size is equal to max capacity
            self.image_stack.remove(self.image_stack[0])  # we will remove the last entry in the stack
            self.image_stack.append(cur_img.copy())  # ans we will append the current image at the front
        elif(size<self.max_size):  # if size is less than maximum capacity
            self.image_stack.append(cur_img.copy())  # we will just append the current image at the front.
            self.stack_undo_index += 1  # and we will update the pointer to current image
        if(len(self.image_stack)>1):   # if length of stack is greater than 1
            self.undo["state"] = NORMAL  # undo button will be enabled
        elif(len(self.image_stack)<=1):  # if it is 1 or 0
            self.undo["state"] = DISABLED  # disable the button
        #print(len(self.image_stack))

    def undo(self):  # this function is for undoing
        size = len(self.image_stack)  # current size of the stack
        if not self.is_original:  # is no image is loaded yet
            messagebox.showwarning(title="Warning", message="No image loaded")  # It will show a warning
        else:  # otherwise
            if size > 1:  # if size is greater than 1
                self.image_stack.pop()  # it will pop out the current image so as to free the stack space
                self.stack_undo_index -= 1  # it will point to the last image before to current image
                self.display_image(self.image_stack[self.stack_undo_index])  # and this will display the image
            else:  # if size<=1
                messagebox.showwarning(title="Warning", message="You can go back up to two images") # there is only image in the stack so it gives warning

    def undo_all(self):  # this function is for performing undo all
        if self.is_original:  # if an image is loaded on to gui

            self.image_stack = [self.original]  # it will empty out the existing stack memory and only stores original.
            self.stack_undo_index = 0  # the pointer will point to the original image
            self.display_image(self.original)  # display the original image
        else:  # if image is not yet loaded
            messagebox.showwarning(title="Warning", message="No image is loaded")  # shows this warning

    def savefile(self):  # this is for saving the image
        disp_img = self.image_stack[self.stack_undo_index]  # this will give the current image on gui
        disp_img = disp_img.astype(np.uint8)  # this will convert all into integers if there are any float
        disp_img= cv2.resize(disp_img,(self.y1_size, self.x1_size))  # resizing to the original shape
        self.rgb1 = Img.fromarray(disp_img)  # converting from array to mat for saving

        hsl = filedialog.asksaveasfile(mode='w', defaultextension=".jpg")  # asking for the path to save
        if hsl is None:  # if no path it just returns
            return
        self.rgb1.save(hsl)  # else it will save the image in the designated path/

    def is_gray(self, imagee):  # to check whether the image is color or gray
        if len(imagee.shape) < 3:  # if shape of image is less than 3
            return True  # then it is a gray image
        if imagee.shape[2] == 1:  # if shape of image is 3 and argument at third place is 1
            return True  # then it is gray
        b, g, r = imagee[:, :, 0], imagee[:, :, 1], imagee[:, :, 2]  # accessing all channels in the image
        if (b == g).all() and (b == r).all():  # if all the channel intensities are equal
            return True  # then it is gray
        return False  # if it doesn't satisy above criteria, then it is color image



    def padding(self, img, pads):  # for zero padding the image
        padding_img = np.zeros((img.shape[0] + 2 * pads, img.shape[1] + 2 * pads), dtype=np.uint8)  # creating the entire shape of image with all zeros
        padding_img[pads:padding_img.shape[0] - pads, pads:padding_img.shape[1] - pads] = img  # only copying the original image part into the above matrix and remaining will be zeros
        return padding_img  # returning the padded image

    def preprocess(self, image):  # before performing transformations some preprocessing
        shape = len(self.image.shape)   # length of the image shape if shape is (200,220,3) length is 3
        pre_image = np.zeros((self.x_size, self.y_size))  # for storing pre processed image

        if(self.bw):  # if image is gray color
            if(len(image.shape)==3):  # if length of shape is 3
                if(image.shape[2]==3):  # if third channel is of length 3
                    pre_image = image[:,:,2]  # we will just store one channel intensity and the new shape will be 2x2
                else:  # if length of 3rd shape is 1 only one intensity is present
                    pre_image = image[:,:,0]  # we will store that intensity and shwpe into 2x2
            else:  # if length of shape is 2, only 2x2 matrix of intensities
                pre_image = image  # we will copy into original image
        else:  # if the image is colour image
            self.hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)  # we will convert into hsv
            pre_image = self.hsv  # and will return hsv image
        return pre_image  # retruning the preprocessed image

    def post_process_gl(self, image, new_intensities):  # post processing for global histogram image
        shape = len(image.shape)  # length of image shape
        if(self.bw):  # if it is gray
            for i in range(self.x_size):  # for each pixel in height
                for j in range(self.y_size):  # for each pixel in width
                    image[i][j] = new_intensities[int(image[i][j])]  # we have to assign new mapped intensity of original intensity
        else:  # if image is color
            for i in range(self.x_size):  # for each pixel in height
                for j in range(self.y_size):  # for each pixel in width
                    image[i][j][2] = new_intensities[int(image[i][j][2])]  # we have to assign new mapped intensity of original intensity
            image = np.array(cv2.cvtColor(image, cv2.COLOR_HSV2RGB))  # we will reconvert the preprocessed image into RGB mode
        return image  # returns pre processed image

    def postprocess(self, image, new_intensities):  # for local histogram, log transform, gamma post processing
        shape = len(image.shape)   # given the length of shape of image, if shape=(20,20,3) length is 3

        if(self.bw):  # if image is gray color
                image = new_intensities  # since in preprocessing we convert 3x3 to 2x2 intensities we will just copy
        else:  # if image is color image
            image[:,:,2] = new_intensities  # we will copy changed intensities to new intensities
            image = np.array(cv2.cvtColor(image, cv2.COLOR_HSV2RGB))  # converting from hsv to rgb
        return image  # returning post processed image

    def global_hist(self):  # for performing global histogram equalisation
        gl_image = self.image_stack[self.stack_undo_index].copy()  # it will retrieve the current image from the stack memory

        pre_gl_image = self.preprocess(gl_image)  # preprocessing the image
        new_gl_image = pre_gl_image.copy()  # copying the preprocessed image to new variable
        if not self.bw:  # if image is color
            new_gl_image = pre_gl_image[:,:,2]  # copy only intensities from preprocessed image
        self.state_disable()  # disabling all buttons while performing transformation
        hist, bin_edges = np.histogram(new_gl_image, bins=256, range=(0, 256))  # histogram of image returns frequency of intensities
        #print(hist)
        hist = np.cumsum(hist)  # performing cumulative sum of the images just like cdf
        #print(hist)
        hist = np.divide(hist, self.x_size * self.y_size)  # dividing the cumulative sum by total no.of pixels
        # print(hist)
        hist = np.multiply(hist, 255)  # multiplying the above by 255
        hist = np.round(hist).astype(int)  # rounding the nwe intensity values
        post_gl_image = self.post_process_gl(pre_gl_image, hist)  # post processing the image
        #cv2.imshow("image",post_gl_image)
        #cv2.waitKey(0)
        self.stack_operation(post_gl_image)  # pushing the image into the stack
        self.state_enable()  # enabling the buttons now
        self.display_image(post_gl_image)  # displaying the image on gui


    def local_histogram(self):  # function for performing local histogram equalisation
        lh_image = self.image_stack[self.stack_undo_index].copy()  # accessing the current image
        self.state_disable()  # disabling the all buttons
        alpha = 0.8  # to account for amount of current image added
        beta = 0.2  # to account for amount of local histogramised image
        hist_pads = 1  # padding taken as 1

        pre_lh_image = self.preprocess(lh_image)  # preprocessing the current image for local hist equalisation
        new_lh_image = pre_lh_image.copy()  # copying to a new variable
        if not self.bw:  # if image is colour image
            new_lh_image = pre_lh_image[:, :, 2].copy()  # copying just the intensities
        new_lh_image = self.padding(new_lh_image, hist_pads)  # zero padding the image
        after_lh_image = np.zeros((self.x_size, self.y_size))  # creating temporary image of original size to store intensities
        for i in range(self.x_size):  # for each pixel in height
            for j in range(self.y_size):  # for each pixel in width
                local_array = new_lh_image[i:i + 3, j:j + 3]  # creating a 3x3 local sliding window
                x_shape = local_array.shape[0]  # height of window
                y_shape = local_array.shape[1]  # width of window
                centre = local_array[1][1]  # central pixel intensity
                # hist, bin_edges = np.histogram(local_array, bins=256, range=(0, 255))
                hist = np.zeros((256, 1))  # creating temporary array for storing 255 intensities frequency
                factor = (x_shape * y_shape)  # total no. of pixels in local window
                for l in range(x_shape):  # for each pixel in height
                    for m in range(y_shape):  # for each pixel in width
                        hist[local_array[l][m]] += 1  # incrementing that intensity
                hist = np.cumsum(hist)  # creating cumulative intensities just like cdf
                intensity = np.round(hist[centre] * (255 / factor))  # multiplying by 255 and dividing by no.of pixels
                intensity = np.round(alpha * centre + beta * intensity).astype(int).astype(int)  # taking alpha portion of original and beta portion of local histogramised image amd rounding

                after_lh_image[i, j] = intensity  # replacing thw central pixel with local histogramised intensity
        local_hist_imag = self.postprocess(pre_lh_image, after_lh_image)  # post processing the resultant image
        self.state_enable()  # disabling the buttons
        self.display_image(local_hist_imag)  # dispalying the image
        self.stack_operation(local_hist_imag)  # pushing into the stack


    def log_transform(self):  # function for logarithmic transformation

        log_image = self.image_stack[self.stack_undo_index].copy()  # accessing the current image from the stack
        self.state_disable()  # disabling all states
        pre_log_image = self.preprocess(log_image)  # preprocessing the image
        new_log_image = pre_log_image.copy()  # copying the preprocessed image
        after_log_image = np.zeros((self.x_size, self.y_size))  # creating temporary array for storing resultant image
        if not self.bw:  # if image is color
            new_log_image = pre_log_image[:, :, 2].copy()  # just storing the intensity values of color
        hist = np.zeros((self.x_size, self.y_size))  # creating tmeporary array for storing intensities for log
        for i in range(self.x_size):  # for each pixel in height
            for j in range(self.y_size):  # for each pixel in width
                hist[i][j] = np.round((255 / math.log10(256)) * math.log10(1 + new_log_image[i][j])).astype(int)  # getting new intensity after transformation
        after_log_image = self.postprocess(pre_log_image, hist)  # postprocessing the image
        self.display_image(after_log_image)  # displaying the image
        self.stack_operation(after_log_image)  # pushing the current image into the stack

        self.state_enable()  # enabling all the buttons

    def get_gamma(self):  # function to get gamma value and perform gamma transformation
        gamma = self.gamma_entry.get()  # getting the user entry value
        if gamma is None:  # if color is image
            messagebox.showwarning(title="Warning", message="Press gamma greater than 0 in the side box")  # show warning and some info
        try:  # whether the user entry is valid or not.
            gamma = float(gamma)  # try to convert to float
            self.gamma_transform(gamma)  # if converted call gammma transformation to perform transformation
        except ValueError:  # if can't convert to float
            messagebox.showwarning(title="Warning", message="Press gamma value greater than 0 in the side box")  # show warning


    def gamma_transform(self, gamma):  # function for gamma transform
        hist = np.zeros((self.x_size, self.y_size))  # creating temp array for storing transformed intensity values
        self.state_disable()  # disabling the buttons
        gamma_image = self.image_stack[self.stack_undo_index].copy()  # accessing current image
        pre_gamma_image = self.preprocess(gamma_image)  # pre process image
        new_gamma_image = pre_gamma_image.copy()  # creating a copy of preprocessed image
        if not self.bw:  # if image is colot
            new_gamma_image = pre_gamma_image[:, :, 2].copy()  # just store intensities
        after_gamma_image = np.zeros((self.x_size, self.y_size))  # create a temporary for storing resultant image

        for i in range(self.x_size):  # for each pixel in height
            for j in range(self.y_size):  # for each pixel in width
                hist[i][j] = round(float(255*(float((new_gamma_image[i][j]/255)**(gamma)))))  # applying transformation and storing new intensities
        after_gamma_image = self.postprocess(pre_gamma_image, hist)  # post processing image
        self.stack_operation(after_gamma_image)  # pust the transforme image into stack
        self.display_image(after_gamma_image)  # display the image
        self.state_enable()  # enabling the buttons

    def negative(self):  # function for negation of image
        self.state_disable()  # disabling the buttons
        neg_image = self.image_stack[self.stack_undo_index].copy()  # accessing the current image
        pre_neg_image = self.preprocess(neg_image)  # preprocessing
        new_neg_image = pre_neg_image.copy()  # creating a copy of original
        if not self.bw:  # if color iamge
            new_neg_image = pre_neg_image[:, :, 2].copy()  # only copy intensities
        after_neg_image = np.zeros((self.x_size, self.y_size))  # create temporary array for storing resultant image
        hist = np.zeros((self.x_size, self.y_size))  # create temporary array for storing resultant intensities
        hist = np.subtract(255, new_neg_image)  # subtracting 255 from original intensities
        after_neg_image = self.postprocess(pre_neg_image, hist)  # post processing image
        self.display_image(after_neg_image)  # displaying image
        self.stack_operation(after_neg_image)  # pushong into the stack
        self.state_enable()  # enabling the states

    def get_filter_size(self):  # getting filter size for blurring
        filter_size = self.blur_entry.get()  # get filter size
        try:  # check if valid
            filter_size = float(filter_size)  # first convert into float, so that "9.0" converts to 9.0
            filter_size = int(filter_size)  # then convert into int
            if(filter_size<=0 or filter_size%2==0):  # if filter size less than 0 or if it is even
                raise ValueError  # raise error
            messagebox.showinfo(title="Help", message="Input positive odd filter size. As gamma increases blur increases")  # show info
            self.blur_operation(filter_size)  # if valid perform blur
        except ValueError:  # catch the error
            messagebox.showwarning(title="Warning", message="Enter positive odd filter size in the entry")  # show warning.
    def blur_operation(self, filter_size):  # function for performing blur
        self.state_disable()  # disabling the buttons
        bl_image = self.image_stack[self.stack_undo_index].copy()  # accessing the current image
        pads = int((filter_size-1)/2)  # from filter size find padding number
        filter = np.ones((filter_size, filter_size))  # initialise the box filter
        norm = np.sum(filter)  # normalising the filter
        pre_bl_image = self.preprocess(bl_image)  # preproxess the image
        new_bl_image = pre_bl_image.copy()  # take acopy of preproxessed image

        if not self.bw:  # if color image
            new_bl_image = pre_bl_image[:,:,2]  # copy only the intensities
        new_bl_image = self.padding(new_bl_image, pads)  # pad the original image
        new_image = np.zeros((self.x_size, self.y_size))  # to store the transformed values
        for i in range(self.x_size):  # for each pixel in height
            for j in range(self.y_size): # for each pixel in width
                result = (filter * new_bl_image[i:i + filter_size, j:j + filter_size]).sum()  # create filtersizexfiltersize array and convolving and then adding them
                result = round(result / norm)  # divide above value by normalisation factor
                new_image[i][j] = result  # store the new transformed intensity in result
        after_bl_image = self.postprocess(pre_bl_image,new_image)  # postproxess the image
        self.display_image(after_bl_image)  # diaplaying the image
        self.stack_operation(after_bl_image)  # push into stack
        self.state_enable()  # enabling the buttons


    def get_sigma(self):  # get varinace from teh user for unsharped masking
        sigma_squared = self.sharp_entry.get()  # getting the value from user
        if sigma_squared is None:  # if no entry
            messagebox.showwarning(title="Warning", message="Enter gaussian variance greater than 0")  # show warning
        try:  # go to catch if error is there
            sigma_squared = float(sigma_squared)  # convert into float
            if(sigma_squared<=0):  # if less than 0
                raise ValueError  # raise error
            messagebox.showinfo(title="Help", message="Input gaussian variance greater than 0. As variance increases sharpening increases")  # help the user with the info
            self.sharpen(sigma_squared)  # perform sharpening
        except ValueError:  # if error exists
            messagebox.showwarning(title="Warning", message="Enter gaussian variance greater than 0")  # show warning

    def gaussian_filter(self, sigma_squared):  # gaussian filter coefficients
        constant = 1  # 1/(2*np.pi*sigma_squared)  # constant values

        constant2 = 1/(2*sigma_squared)  #  another constant
        x = np.array([[-2,-1,0,1,2],[-2,-1,0,1,2],[-2,-1,0,1,2], [-2,-1,0,1,2],[-2,-1,0,1,2] ])  # diaplaying x positions
        y = np.array([[-2,-2,-2,-2,-2],[-1, -1, -1, -1, -1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [2,2,2,2,2]])  # dispalying y positions
        gaus_filter = np.zeros((5,5))   # creating temporary array for filtering
        for i in range(5):  # for each x-valoue in filter
            for j in range(5):  # for each y-value in filter
                gaus_filter[i][j] = math.exp(-((x[i][j])*(x[i][j])+(y[i][j])*(y[i][j]))*constant2)  # gaussian filter formula
        gaus_filter = (np.multiply(gaus_filter, constant))  # multiply ablove by constant
        return gaus_filter  # return the filter
    def sharpen(self, variance):  # function for performing sharpening
        gauss_filter = self.gaussian_filter(variance)  # get the gaussina filter
        #print(gauss_filter)
        filter_size = gauss_filter.shape[0]  # getting filter size
        alpha = 0.2  # getting alpha for combining unsharp mask
        self.state_disable()  # disabling all the states
        pads = 2  # number of paddings
        sh_image = self.image_stack[self.stack_undo_index].copy()  # accessing the current image
        norm = np.sum(gauss_filter)  # if posstible normalise the filter
        print(norm)
        if norm==0:  # if norm is zero
            norm=1  # then norm = 1
        pre_sh_image = self.preprocess(sh_image)  # preprocess image
        new_sh_image = pre_sh_image.copy()  # copy the preprocessed image

        if not self.bw:  # if color image
            new_sh_image = pre_sh_image[:, :, 2]  # copy only intensities portion
        new_sh_image = self.padding(new_sh_image, pads)  # zero padding the image
        new_image = np.zeros((self.x_size, self.y_size))  # create temporaray array for storing transformed values
        temp_image = new_image.copy()  # just to store masked values for checking
        for i in range(self.x_size):  # for each pixel in height
            for j in range(self.y_size):  # for each pixel in width
                result = (gauss_filter * new_sh_image[i:i + filter_size, j:j + filter_size]).sum()  # performing convolution and then summing
                result = round(result / norm)  # divide by norm
                new_image[i][j] = result  # store into newimage
                new_image[i][j] = new_sh_image[i][j] - new_image[i][j]  # mask = original image - blurred image
                temp_image[i][j] = new_image[i][j]  # storing mask values
                new_image[i][j] = new_sh_image[i][j] + alpha*new_image[i][j]  # result image = original image + alpha * mask

        new_image = ((new_image - new_image.min()) * (1 / (new_image.max() - new_image.min()) * 255)).astype('uint8')  # if pixels are out of range then bring into the range
        after_sh_image = self.postprocess(pre_sh_image, new_image)  # perform post processing
        self.display_image(after_sh_image)  # diaplying the original image
        #self.display_image(temp_image)  # for displaying gaussian mask
        self.stack_operation(after_sh_image)  # push the modified image into the stack
        self.state_enable()  # enable the buttons


root = Tk()  # instantiating the main gui
root.resizable(0, 0)  # this function disables maximising the gui
root.title("Basic Image Editor")  # title for the gui
root.geometry("800x700")  # fix the geometry
root.configure(bg="blue")  # background color of gui

temp = Image_editor(root)  # instantiating the object of image editor class with root frame as parameter.

root.mainloop()  # This is like a while loop which goes on executing until some interruption happens in the gui.
