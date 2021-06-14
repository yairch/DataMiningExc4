from tkinter import Tk, Label, Button, Entry, END, W, IntVar
from tkinter import messagebox
import tkinter as tk
import os
import naiveBayesModel


class naiveBayesGUI:

    def __init__(self, master):
        self.path = ""
        self.valid_path = False
        self.valid_bins = False
        self.num_of_bins = None
        self.feature_structure = None
        self.model = None

        # define window size
        canvas = tk.Canvas(master, width=300, height=100)
        canvas.grid(columnspan=3)

        self.master = master  # tkinter root
        master.title("Naive Bayes Classifier")  # app title

        # working directory path
        self.path_label = Label(master, text="Directory Path")
        register_path = master.register(self.set_path)  # we have to wrap the command
        self.path_entry = Entry(master)
        self.path_entry.config(validate="key", validatecommand=(register_path, '%P'))
        self.browse_button = Button(master, text="Browse", command=lambda: self.on_click_browse())

        # number of bins input
        self.binning_label = Label(master, text="Discretization Bins")
        register_binning = master.register(self.set_num_of_bins)  # we have to wrap the command
        self.binning_entry = Entry(master)
        self.binning_entry.config(validate="key", validatecommand=(register_binning, '%P'))

        # model functionality
        self.test_classification_button = Button(master, text="Classify", command=lambda: self.on_click_classify())
        self.test_classification_button['state'] = 'disabled'

        self.train_model_button = Button(master, text="Build", command=lambda: self.on_click_build())
        self.train_model_button['state'] = 'disabled'

        # design
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
            self.train_model_button['state'] = 'disabled'
        else:
            try:

                files = os.listdir(self.path)
                if train_filename in files and test_filename \
                        in files and structure_filename in files:
                    self.valid_path = True
                    messagebox.showinfo(
                        title="Naive Bayes Classifier",
                        message="Valid path has been set")

                    if self.valid_path and self.valid_bins:
                        self.train_model_button['state'] = 'normal'

                    return True
                else:
                    messagebox.showerror(
                        title="Naive Bayes Classifier",
                        message="Error occurred when opening directory. Not all needed files present")
                    self.path_entry.delete(0, END)
                    self.train_model_button['state'] = 'disabled'
                    return False
            except:
                messagebox.showerror(
                    title="Naive Bayes Classifier",
                    message="Error occurred when opening directory. No directory named {0}".format(self.path))
                self.path_entry.delete(0, END)
                self.train_model_button['state'] = 'disabled'
                return False

    def set_num_of_bins(self, bins):
        """
        :param bins: number of bins as chosen by user
        :return: boolean value indicating validity on bins number
        """
        if bins == "":  # no number of bins has been entered
            self.num_of_bins = 0
            self.train_model_button['state'] = 'disabled'
            return True
        else:
            try:
                bins = int(bins)
                if bins > 0:
                    self.num_of_bins = bins
                    self.valid_bins = True

                    if self.valid_path and self.valid_bins:
                        self.train_model_button['state'] = 'normal'

                    return True
                else:
                    return False
            except ValueError:
                return False

    def on_click_build(self):

        if self.num_of_bins < 2:
            messagebox.showerror(
                title="Naive Bayes Classifier",
                message="Bin amount must be at least 2")
            return False

        # load required files
        train_set = naiveBayesModel.import_data(self.path, 'train.csv')
        self.feature_structure = naiveBayesModel.read_structure(self.path, 'Structure.txt')

        if train_set is False or not self.feature_structure:
            messagebox.showerror(
                title="Naive Bayes Classifier",
                message="Either train set, test set, or structure file are empty")
            return False

        self.model = naiveBayesModel.build_model(train_set=train_set,
                                                 feature_structure=self.feature_structure,
                                                 n_bins=self.num_of_bins)
        messagebox.showinfo(
            title="Naive Bayes Classifier",
            message="Building classifier using train-set is done!")

        self.test_classification_button['state'] = 'normal'
        return True

    def on_click_classify(self):

        # load test file
        if not self.path or not self.feature_structure or not self.model:
            messagebox.showerror(
                title="Naive Bayes Classifier",
                message="Model haven't been built")
            return False

        test_set = naiveBayesModel.import_data(self.path, 'test.csv')

        if test_set is False:
            messagebox.showerror(
                title="Naive Bayes Classifier",
                message="Test set is empty")
            return False

        predictions = naiveBayesModel.classify(model=self.model,
                                               test_set=test_set,
                                               feature_structure=self.feature_structure,
                                               n_bins=self.num_of_bins)

        naiveBayesModel.save_predictions(path=self.path,
                                         predictions=predictions)
        messagebox.showinfo(
            title="Naive Bayes Classifier",
            message="Classified successfuly and saved predictions. Press OK to exit.")
        self.master.destroy()
        return True

root = Tk()
my_gui = naiveBayesGUI(root)
root.mainloop()
