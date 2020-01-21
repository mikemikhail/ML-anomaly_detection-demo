# -*- mode: ruby -*-
# vi: set ft=ruby :

# All Vagrant configuration is done below. The "2" in Vagrant.configure
# configures the configuration version (we support older styles for
# backwards compatibility). Please don't change it unless you know what
# you're doing.
Vagrant.configure("2") do |config|
  # The most common configuration options are documented and commented below.
  # For a complete reference, please see the online documentation at
  # https://docs.vagrantup.com.

  # Every Vagrant development environment requires a box. You can search for
  # boxes at https://vagrantcloud.com/search.
  config.vm.define "TICK" do |node|
      node.vm.box = "kuhlskev/Telemetry_Collection_Stack"

      node.vm.network :forwarded_port, guest: 22, host: 2222, id: 'ssh', auto_correct: true
      # Forward API Ports
      node.vm.network :forwarded_port, guest: 8088, host: 8088, id: '8088', auto_correct: true
      node.vm.network :forwarded_port, guest: 8888, host: 8888, id: 'chronograf', auto_correct: true
      node.vm.network :forwarded_port, guest: 9200, host: 9200, id: '9200', auto_correct: true
      node.vm.network :forwarded_port, guest: 9300, host: 9300, id: '9300', auto_correct: true
      node.vm.network :forwarded_port, guest: 3000, host: 3000, id: 'Grafana', auto_correct: true
      node.vm.network :forwarded_port, guest: 8068, host: 8068, id: 'InfluxDB', auto_correct: true
      node.vm.network :forwarded_port, guest: 5901, host: 5901, id: 'VNC', auto_correct: true
  # Disable automatic box update checking. If you disable this, then
  # boxes will only be checked for updates when the user runs
  # `vagrant box outdated`. This is not recommended.
  # config.vm.box_check_update = false
  config.ssh.insert_key = false

  # Share an additional folder to the guest VM. The first argument is
  # the path on the host to the actual folder. The second argument is
  # the path on the guest to mount the folder. And the optional third
  # argument is a set of non-required options.
     node.vm.synced_folder ".", "/home/vagrant/tecdev-2765"

  # Provider-specific configuration so you can fine-tune various
  # backing providers for Vagrant. These expose provider-specific options.
  # Example for VirtualBox:
  #
     node.vm.provider "virtualbox" do |vb|
  #   # Display the VirtualBox GUI when booting the machine
       #vb.gui = true
  #
  #   # Customize the amount of memory on the VM:
       vb.memory = "4096"
       vb.customize ["modifyvm", :id, "--uart1", "0x3f8", "4"]
       vb.customize ["modifyvm", :id, "--uartmode1", "file", File.join(Dir.pwd, "TICK/serial_port1")]
     end
  #
  end
  # View the documentation for the provider you are using for more
  # information on available options.

  # Enable provisioning with a shell script. Additional provisioners such as
  # Puppet, Chef, Ansible, Salt, and Docker are also available. Please see the
  # documentation for more information about their specific syntax and use.
  # config.vm.provision "shell", inline: <<-SHELL
  #   apt-get update
  #   apt-get install -y apache2
  # SHELL
end
