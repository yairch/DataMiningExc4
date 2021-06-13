from tkinter import Tk, Label, Button, Entry, END, W
from tkinter import messagebox
import tkinter as tk
import os


class naiveBayesGUI:

    def __init__(self, master):
        self.path = ""
        self.num_of_bins = None

        canvas = tk.Canvas(master, width=300, height=100)
        canvas.grid(columnspan=3)

        self.master = master
        master.title("Naive Bayes Classifier")

        self.path_label = Label(master, text="Directory Path")

        register_path = master.register(self.set_path)  # we have to wrap the command
        self.path_entry = Entry(master)
        self.path_entry.config(validate="key", validatecommand=(register_path, '%P'))

        self.browse_button = Button(master, text="Browse", command=lambda: self.on_click_browse())

        self.binning_label = Label(master, text="Discretization Bins")

        self.test_classification_button = Button(master, text="Classify", command=lambda: self.on_click_classify())
        self.train_model_button = Button(master, text="Build", command=lambda: self.on_click_build())

        register_binning = master.register(self.set_num_of_bins)  # we have to wrap the command
        self.binning_entry = Entry(master)
        self.binning_entry.config(validate="key", validatecommand=(register_binning, '%P'))

        self.path_entry.grid(row=0, column=1, sticky=W)
        self.binning_entry.grid(row=1, column=1, sticky=W)

        self.path_label.grid(row=0, column=0, sticky=W)
        self.binning_label.grid(row=1, column=0, sticky=W)

        self.browse_button.grid(row=0, column=2, columnspan=2, sticky=W)
        self.train_model_button.grid(row=2, column=1)
        self.test_classification_button.grid(row=3, column=1)

    def set_path(self, path):
        self.path = path
        return True


    def on_click_browse(self):
        """
        This function checks the validity of a folder path and the
        presence of each of the needed files: test.csv, train.csv, Structure.txt
        :return: return boolean value indicating the validity of the given folder
        """
        train_filename = "train.csv"
        test_filename = "test.csv"
        structure_filename = "Structure.txt"

        if self.path == "":
            messagebox.showerror(
                title="Naive Bayes Classifier",
                message="Error occurred when opening directory, no path specified.")
        else:
            try:

                files = os.listdir(self.path)
                if train_filename in files and test_filename \
                        in files and structure_filename in files:
                    return True
                else:
                    messagebox.showerror(
                        title="Naive Bayes Classifier",
                        message="Error occurred when opening directory. Not all needed files present")
                    self.path_entry.delete(0, END)
                    return False
            except:
                messagebox.showerror(
                    title="Naive Bayes Classifier",
                    message="Error occurred when opening directory. No directory named {0}".format(self.path))
                self.path_entry.delete(0, END)
                return False

    def set_num_of_bins(self, bins):
        """
        :param bins: number of bins as chosen by user
        :return: boolean value indicating validity on bins number
        """
        if bins == "":  # no number of bins has been entereds
            messagebox.showerror(
                title="Naive Bayes Classifier",
                message="No number of bins specified")
            self.binning_entry.delete(0, END)
            return False
        else:
            try:
                bins = int(bins)
                if bins > 0:
                    self.num_of_bins = bins
                    return True
                else:
                    messagebox.showerror(
                        title="Naive Bayes Classifier",
                        message="Invalid number of bins: {0}".format(bins))
                    self.binning_entry.delete(0, END)
                    return False
            except ValueError:
                self.binning_entry.delete(0, END)
                messagebox.showerror(
                    title="Naive Bayes Classifier",
                    message="Invalid number of bins: {0}".format(bins))
                return False

    def on_click_build(self):

        if self.num_of_bins:
            pass
        # call function that loads the training set and trains the model
        pass

    def on_click_classify(self):

        # call the function that loads the test file and classifies the entries
        pass


root = Tk()
my_gui = naiveBayesGUI(root)
root.mainloop()
