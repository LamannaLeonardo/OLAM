# Online Learning of Action Models (OLAM) for classical planning 
This repository contains the official code of the OLAM algorithm presented at IJCAI 2021, for details about the method please see the [paper](https://www.ijcai.org/proceedings/2021/0566.pdf).


## Installation
The following instructions have been tested on Ubuntu 20.04.


1. Create a Python 3.9 virtual environment using conda or pip.
```
 conda create -n olam python=3.9
```
2. Activate the environment and install dependencies
```
 conda activate olam
 pip install numpy pandas scipy matplotlib xlwt xlrd
```
3. Install Oracle Java SE Development Kit (jdk), this should be a private installation that does not interfere with already existing ones. For downloading the jdk archive you can refer to this [link](https://www.oracle.com/java/technologies/downloads/). After downloading the jdk, move it into the "Java" folder and extract the archive by executing the command ``` tar -xzvf jdk-XX_linux-x64_bin.tar.gz ``` where XX is the jdk version you downloaded. For example, download the archive jdk-17_linux-x64_bin.tar.gz, move it into the "Java" folder, extract the archive with ``` tar -xzvf jdk-17_linux-x64_bin.tar.gz ``` and then delete the unnecessary file with ``` rm  jdk-17_linux-x64_bin.tar.gz```.

4. Following instructions provided in the [offical FastDownward site](https://www.fast-downward.org/), download FastDownard in a directory called "FD", move the directory "FD" into the "Planners" directory, go into the subdirectory "Planners/FD" and compile FastDownward through the command ```./build.py```.

5. From the [offical FastForward site](https://fai.cs.uni-saarland.de/hoffmann/ff.html), download FF-v2.3.tgz (you can directly download it from this [link](https://fai.cs.uni-saarland.de/hoffmann/ff/FF-v2.3.tgz)), move it into the "Planners/FF" directory, extract the archive ```tar -xf FF-v2.3.tgz```, go into the installation directory with ```cd FF-v2.3``` and compile FastForward with ```make```. Finally move the "ff" executable in the parent directory through the command ```mv ff ../```, go to the parent directory ```cd ../``` and delete unnecessary files with ```rm -r FF-v2.3``` and ```rm FF-v2.3.tgz```.

6. Verify everything is working properly by executing ```python main.py -d blocksworld``` (the execution may take few minutes).


## Execution

### Running OLAM
By default, the OLAM algorithm is run over all instances of all domains in the "Analysis/Benchmarks" directory. If you want to run OLAM on a single domain use the ```-d``` or ```--domain``` argument. For example, to run OLAM on the blocksworld domain execute ```python main.py -d blocksworld```.

### Running OLAM on custom problems
To change the problem instances, add/remove pddl files contained into the domain directory. For instance, to run OLAM only on the problem instance "1_p00_blocksworld_gen.pddl" of the blocksworld domain, remove all problems but "1_p00_blocksworld_gen.pddl" in the directory "Analysis/Benchmarks/blocksworld".
Similarly you can add new problem instances.

### Running OLAM on custom domains
To add a new classical planning domain "mydomain", add the pddl domain file "mydomain.pddl" into the "Analysis/Benchmarks" directory, then create the subdirectory "Analysis/Benchmarks/mydomain" with the problem instances. Finally execute the command ```python main.py -d mydomain```.


## Log and results
When you run OLAM, a new directory (e.g. "run_0") with all logs and results is created in the "Analysis" folder. The created directory contains three directories:
1. Tests: contains one directory for each considered domain, e.g. "Analysis/run_0/Tests/blocksworld", which contains one directory for each problem instance. The action model <img src="https://render.githubusercontent.com/render/math?math=M"> learned after the resolution of the problem is called "domain_learned_certain.pddl", while "domain_learned.pddl" is the action model <img src="https://render.githubusercontent.com/render/math?math=M^-_?"> including also the uncertain negative effects (see the [paper](https://www.ijcai.org/proceedings/2021/0566.pdf) for further details)
2. Results_cert: contains one excel file for each domain with the metrics reported in the [paper](https://www.ijcai.org/proceedings/2021/0566.pdf), the evaluation consider the model <img src="https://render.githubusercontent.com/render/math?math=M"> without uncertain negative effects (i.e. "domain_learned_certain.pddl")
3. Results_uncert_neg: contains one excel file for each domain with the metrics reported in the [paper](https://www.ijcai.org/proceedings/2021/0566.pdf), the evaluation consider the model <img src="https://render.githubusercontent.com/render/math?math=M^-_?"> with uncertain negative effects (i.e. "domain_learned.pddl")


## Citations
If you find OLAM useful, please cite this paper:
```
@inproceedings{ijcai2021-566,
  title     = {Online Learning of Action Models for PDDL Planning},
  author    = {Lamanna, Leonardo and Saetti, Alessandro and Serafini, Luciano and Gerevini, Alfonso and Traverso, Paolo},
  booktitle = {Proceedings of the Thirtieth International Joint Conference on
               Artificial Intelligence, {IJCAI-21}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Zhi-Hua Zhou},
  pages     = {4112--4118},
  year      = {2021},
  month     = {8},
  note      = {Main Track}
  doi       = {10.24963/ijcai.2021/566},
  url       = {https://doi.org/10.24963/ijcai.2021/566},
}
```
