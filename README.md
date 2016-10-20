SciLuigi CaseStudy Virtual Machine
==================================

This repository contatins the code to set up a virtual machine with the case
study workflow for SciLuigi, runnable from within a Jupyter Notebook. 

It does also include the SciLuigi case study workflow, which can be found
in [this folder](https://github.com/pharmbio/bioimg-sciluigi-casestudy/tree/master/roles/sciluigi_usecase/files/proj/largescale_svm).

For a view-only version of the Jupyter Notebook with the full case study workflow,
[see here](https://github.com/pharmbio/bioimg-sciluigi-casestudy/blob/master/roles/sciluigi_usecase/files/proj/largescale_svm/wffindcost.ipynb).

Screenshot
----------

![Screenshot](http://i.imgur.com/SbxZxiP.png)

Usage
-----

There are two ways to get and start using this virtual machine. You can either
download a pre-made virtual machine image in the open virtualization format
(.ova) that can be imported in a virtual machine player such as VMWare player
or VirtualBox client.

But you can also build the image from scratch, by using the code in this github
repository, given that you have installed the dependencies (Vagrant, Ansible
and VirtualBox).

### Using the Pre-made Virtual Machine Image

To install and use the pre-made virtual machine, follow the steps below:

Usage: 

1. Import the .ova image into a Virtual Machine software such as Virtual box. 
2. Start the virtual machine. 
3. Log in with ubuntu and changethis...
4. Open a terminal and execute the passwd command, to immediately set a new password.
5. Click the "Open Jupyter Notebook" icon on the desktop.
6. Inside Jupyter, click: Cell > Run all cells
7. The workflow will now start.

### Building the virtual machine from scratch

- (TBC)
