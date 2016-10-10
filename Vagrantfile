# -*- mode: ruby -*-
# vi: set ft=ruby :

VAGRANTFILE_API_VERSION = "2"

Vagrant.configure(VAGRANTFILE_API_VERSION) do |conf|

  # --------------------------------------------------------------------
  # Definitions for the VirtualBox machine
  # --------------------------------------------------------------------
  conf.vm.define "bioimg", autostart: true do |bimg|
    bimg.vm.provider "virtualbox" do |vbox|
      vbox.memory = 2048
      vbox.cpus = 2
      #vbox.gui = true
    end
    bimg.vm.box = "ubuntu/xenial64"
    bimg.vm.network "forwarded_port", guest: 80, host: 8080
    bimg.vm.provision :ansible do |ansible|
      ansible.playbook = "playbook.yml"
    end
    # Tell the user what to do next
	bimg.vm.provision "shell", inline: "echo 'Finished! Now try logging in with: vagrant ssh virtualbox'"
  end
end
