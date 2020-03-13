+++
title = "Ansible tutorail for human"
[taxonomies]
categories = ["Technology", "DevOps"]
+++

We need documents for human, not for robots.

<!-- more -->

> What is ansible?

Ansible is a tool for DevOps. It can be used to：

* define server groups in inventory
* control a group of servers like localhost
* setup environments on servers
  * install, update, remove apps
  * write, copy files
* execute command on servers
* use playbooks to execute a series of tasks
* use others' playbooks as roles from ansible galaxy

## Tutorial

The easiest way to learn how to use ansible is learning from examples. Only when you find some key words that confuse you that you need to check the official documents with search engines.

Here I'd like to give you some simple introductions.

### Inventory

TOML like file that defines the group of servers. If you already defined some hosts in `$HOME/.ssh/config`, you can use the host name instead of IP address.

```toml
[master]
10.0.0.1

[worker]
10.0.10.1
10.0.10.2
10.0.10.3
```

In the file above, we define two groups of servers. Then you can use `master`, `worker`, `all` as group name. Save this file as `/etc/ansible/hosts` so you can use it anywhere.

### Ansible cmd

Ping all the servers: `ansible all -m ping`.

Run a live command: `ansible worker -a "/bin/echo $USER"`

### Playbooks and roles

You can execute a series of tasks defined in the playbooks. First of all, check if there exists roles that already done the same thing you need. Roles can be found from [Ansible Galaxy](https://galaxy.ansible.com/).

Playbooks is defined in yaml file.

```yml
- hosts: all
  become: yes
  roles:
    - nvidia.nvidia_driver
    - nvidia.nvidia_docker
  tasks:
    - name: turn off swap
      command: swapoff -a
      when: ansible_swaptotal_mb > 0

    - name: add k8s GPG key
      apt_key: url=https://packages.cloud.google.com/apt/doc/apt-key.gpg
      
    - name: install kubectl kubelet kubeadm
      apt:
        name: ['kubelet','kubectl','kubeadm']
        state: present
        update_cache: yes
```

For each task, you can do almost everything like `command`, `apt`, `copy`, `replace`. Find what you need with search engines.

Execute playbooks:

```sh
ansible-playbook -i inventory k8s.yml
```

## Finally

This is almost everything you need to know about ansible as a beginner :)

As usual, GitHub and StackOverflow are still your best friends.