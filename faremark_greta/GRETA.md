# greta summer@epfl 2026 - watermarking and free-rider detection in decentralized federated learning

## project description
The overall project goal is to analyse and find a solution for free-rider detection in decentralized federated learning.


**Free-rider detection in decentralized learning models using watermarks**
Decentralized learning has emerged as a promising alternative to centralized and federated paradigms for training machine learning models without relying on a central server. In fully decentralized settings, nodes collaboratively optimize a shared objective through local computations and peer-to-peer communication. However, free-rider behavior remains a critical challenge: some nodes may benefit from the global model while contributing low-quality updates, random gradients, stale parameters, or no meaningful computation at all. Such behavior can degrade convergence, compromise fairness, and undermine trust in the system. This project aims to design a robust watermark-based accountability framework to detect, quantify, and mitigate free-riding in decentralized learning systems. The core idea is to leverage model watermarking techniques to embed verifiable signals into the training process, enabling each client to claim legitimate safeguarding of intellectual property rights of the FL models. 
Research questions: – Can watermarking techniques developed in federated learning (e.g., [1]) be adapted to fully decentralized settings without a central coordinator? – How robust are watermark-based detection mechanisms against adversarial behaviours, such as collusion, gradient manipulation, or attempts to forge the watermark? 

To contribute effectively to this project, we highly value:
* Strong ML fundamentals and proficiency in ML implementation
* Strong mathematical foundation and interest in probability theory, algebra, and analysis 
[1] Li, Li, Xinpeng Zhang, Hanzhou Wu, Guorui Feng, and Weiming Zhang. “FareMark: Model-Watermark-Driven Free-Rider Detection in Federated Learning Model.” IEEE Internet of Things Journal (2025)



## sections
> 1. [updates](#updates) and [plan](#plan)
> 2. [resources](#resources) - [useful-commands](#daily-usage-of-server)
> 3. [code-documentation](#my-code)
> 4. [results](#results)

---
---

## updates
| Date | Updates | Notes |
|------|-------|-------|
| June 2 | [x] brainstorm session  | - |
| June 9 | [x] initial code exploration <br> [x] initial concepts and FareMark paper review | - |
| June 11 | [1] check which papers cite FareMark <br> [x] [paper deep dive](FareMark.md) and watermarking procedure <br> [x] potential issues for DFL vs. FL <br> [2] trigger classes (do they need to be unique for each client) <br> [3] trigger class weaknesses | [1] only 2 papers cite it. they talk about [AIIP-Chain: Fair Copyright Sharing With Credible Ownership Verification in AI Model Trading](https://ieeexplore.ieee.org/abstract/document/11239438) (brief mention of watermarking as a method to detect free-riders) and [Intellectual property protection for deep learning model and dataset intelligence](https://www.sciencedirect.com/science/article/pii/S0952197625030556#b64) (table 7 quick mention) <br> [2] best case scenario yes (server just stores the class label at verification and picls any images in the class to verify). in case there are more, the empirical data shows that it's fine and the server just pre-specify and stores the exact imaes used by each client (storage increase). **potential better solution**: different paritition based on features instead <br> [3] **potential issue 1**: partial free-rider attack by only training the trigger classes + trigger class needs to remain the same throughout training and testing - **potential issue 2**: mainly for DFL, dynamic client participation |
| June 16 | [X] emailed Xinpeng Zhang and Li Li for code <br> [X] basic re-implementation using Claude | - |
| June 23 | [x] build basic federated learning framework <br> [1] test to make sure everything is correct <br> [] document and present <br> [2] build the free-rider attacks <br> [x] build the watermarking algorithm <br> [3] test and validate everything is correct and matches the paper <br> [] document + double check with paper + present | [1] stage 1 tests: smoke test good + CIFAR-10 baseline (just FL) good + ResNet-18/MNIST (just FL) good <br> [2] stage 2 tests: smoke test good + prev_attack good + gaussian_noise attacks good -> have to show decline <br> [3] stage 3 tests: smoke test + watermarking algorithm + stage 4 tests <br> [] test and run experiments from the paper |

---

## plan
| Date | Tasks |
|------|-------|
| June 2 | [x] brainstorm ideas |
| June 9 | [x] explore codebase and understand the framework (see Milos for setup and help) <br> [x] read and review FareMark paper |
| June 16 | [X] setup GPU clusters (Milos instructions) <br> [] get Claude pro |
| June 23 | [] implement the FareMark paper and reproduce the results <br> [] run all basic experiments from the paper and obtain proof that code is good <br> [] deep dive into code - documentation and compare with algorithm in paper to make sure everything is correct <br> [] short presentation for JSM  to prove everything is working <br> [] deep dive into the paper and code |
| July 2 | [] send a follow up email to authors <br> [] next steps for the project ? |
| July 7 | [] |
| July 14 | [] |
| July 21 | [] start writing report ? |
| July 28 | [] |
| August 4 | [] |
| August 11 | [] |
| August 18 | [] |
| August 25 | [] |


---
### submissions
| Date | Tasks |
|------|-------|
| August 29 | - last day |


### NOTES/questions
- graph colouring - number of nodes and number of colours = number of unique classes needed for watermarking
- federated learning but no data privacy ?
- goal: attack method that utilizes the least amount of resources (eg. only train on the trigger class) to be a free-rider and then test the detection method on it that based on watermarking in outer layer. no matter the data boundary, the free-rider will be detected.
- collusion attack ? every random amout of rounds ? neighbouring clients ?
- penalty for free-riders ? how to mitigate them ?
- facking fairness paper ? optimal transport ?
- attack: threshold is averaged

- NOTE FOR DATA PARTITIONING
- NOTE FOR calibrating n threshold
- QUESTION: more free-riders - weaker embedding for honest minority clients ?

---
---
## resources
#### my code
- [forked-repository](https://github.com/zu-greta/decentralizepy)
    - FareMark reproduction code and my own implementation of the paper can be found under the `faremark_greta` folder. The code is structured in stages and each stage has a correctness gate to ensure that the foundation is sound before building on it. The code is modular and can be ported into decentralizepy later if we need genuinely distributed runs.
- [original-code](https://github.com/sacs-epfl/decentralizepy) - original code from the SaCS lab for decentralized federated learning

- TODO add instructions for usage and documentation for the code in the readme of the forked repo

Structure: TODO
```
decentralizepy
.
├── README.rst                              # setup for the framework and instructions for usage
├── ...
├── watermarking_freerider
│   ├── watermarking_freerider              # folder containing all my code 
│   └── ... TODO
└── ...
```

---
#### server access 
- contact Milos to get access to the RCP group
- setup your acount using the instructions provided by Milos and the [environment preparations wiki page](https://wiki.rcp.epfl.ch/home/CaaS/FAQ/how-to-prepare-environment). then use the following [runai wiki pages](https://wiki.rcp.epfl.ch/home/CaaS/FAQ/how-to-use-runai). make sure you are on EPFL wifi or VPN to access the pages and server.
- examples of setup for Dockerfile, build.sh, requirements.txt can be found in the `decentralizepy/faremark_greta/infra` folder. you can use them as a reference to setup your own environment for the project. make sure to replace the commands and configurations with your own.

- RCP registry can be found at [https://registry.rcp.epfl.ch](https://registry.rcp.epfl.ch)
- RunAI can be found at [runai sso login](https://app.run.ai/auth/realms/rcpepfl/protocol/openid-connect/auth?response_type=code&connection=rcpepfl&client_id=runai-admin-ui&redirect_uri=https%3A%2F%2Frcpepfl.run.ai%2Flogin%2Fcallback&scope=openid+email+profile&state=954d72dc-49fb-4c91-a24e-a45293f69120&code_challenge=XCX_JlXjQ6QSNr22QkiK9z2cQcXWjDaxROlSagGWeAU&code_challenge_method=S256)

- Jumphost access: `ssh <username>@haas001.rcp.epfl.ch -o PubkeyAuthentication=no` and enter your EPFL password when prompted. From the jumphost, create your persistent storage directory: `mkdir -m /mnt/sacs/scratch/home/<username>`
- From the jumphost you can also find your UID and GID using the command `id -u` and `id -g`. You will need these to set up your RunAI account.

---
#### daily usage of server 
- `ssh <username>@haas001.rcp.epfl.ch -o PubkeyAuthentication=no` and enter your EPFL password when prompted
- `cd /mnt/sacs/scratch/home/<username>` to access your persistent storage directory
- `git clone <your-forked-repo-url>` to clone your forked repo in the server or `git pull` to update it if you already have it cloned
- `cd <your-repo-name>` to access the code

- `watch nvidia-smi` to monitor memory and power usage during run
- `sftp` to dowload large files from the server
- ui.perfetto.dev to view trace of runs (eg. `/mnt/nobackup/omicha1/msc-research-exploration/energy_effiency/training-trace-fmoe-128-9-rank-0.json`)

- runai commands:
    - `runai submit job <job-name> --image <image-name> --gpu 1 --cpu 4 --memory 16Gi --command "bash run.sh"` to submit a job
    - `runai list jobs` to list all jobs
    - `runai logs <job-name>` to view logs of a job
    - `runai delete job <job-name>` to delete a job
- kubectl commands:
    - `kubectl get pods` to list all pods
    - `kubectl logs <pod-name>` to view logs of a pod
    - `kubectl delete pod <pod-name>` to delete a pod
    - `kubectl logs -n runai-sacs-zu <pod-name> -f` to view logs of a pod in real-time
---
#### forking repo
fork the repo and clone it. then:
- `git remote add upstream git@github.com:<repository-name>.git`
- `git remote -v`

to sync it with the original repo
- `git fetch upstream`
- `git checkout main`
- `git merge upstream/main`
- `git push origin main`
---
#### merging from other branch
- `git pull` from both current and other branch so that you are up to date
- `git checkout <CURRENTBRANCH>`
- `git fetch origin <OTHERBRANCH>`
- `git merge origin/<OTHERBRANCH>`
---
#### scp to download to local
- `scp -r <source-path> /Users/gretazu/Downloads`
- `scp -r zu@haas001.rcp.epfl.ch:/mnt/sacs/scratch/home/zu/<result-path> .`
---
#### tmux
- `tmux new -s <SESSION NAME>`
- run script
- ctrl b
- d
- `tmux ls`
- `tmux attach -t <SESSION NAME>`
- once done: `tmux kill-session -t <SESSION NAME>`

---
---
## results
1. code
    - code runs
    - structured for future usage
    - documented (comments and readme)
    - leave notes for usage and future work
    - results folder with logs and plots 
2. report/paper
    - ?
3. presentation 
    - ?