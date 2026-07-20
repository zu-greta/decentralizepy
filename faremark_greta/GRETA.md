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
| June 23 | [x] build basic federated learning framework <br> [1] test to make sure everything is correct <br> [] document and present <br> [2] build the free-rider attacks <br> [x] build the watermarking algorithm <br> [3] test and validate everything is correct and matches the paper <br> [x] document + double check with paper + present | [1] stage 1 tests: smoke test good + CIFAR-10 baseline (just FL) good + ResNet-18/MNIST (just FL) good <br> [2] stage 2 tests: smoke test good + prev_attack good + gaussian_noise attacks good -> have to show decline <br> [3] stage 3 tests: smoke test + watermarking algorithm + stage 4 tests <br> [] test and run experiments from the paper |
| July 2 | [x] paper experiments reproduced <br> [1] new attacks basic run | [1] things tried: non-iid, threshold testing, mixed attack based on trigger only + common samples |
| July 7 | [1] no working results yet - needs more tuning for the new attacks | [1] testing how much training is needed to start with (cannot just do trigger samples, need a full shard to warm up) <br> testing some autopilot dynamic way |
| July 16 | [x] threshold and different knobs experiments for submarine attack to be refined | - |
| July 21 | [] threshold fixed <br> [] baseline submarine attack results for iid, full scope, tap/coast, and +5/common <br> [] have basic plots for results - just prove that on iid, with the harsh threshold, free-riding is possible with either tap/coast or +5/common | - |

---

## plan
| Date | Tasks |
|------|-------|
| June 2 | [x] brainstorm ideas |
| June 9 | [x] explore codebase and understand the framework (see Milos for setup and help) <br> [x] read and review FareMark paper |
| June 16 | [X] setup GPU clusters (Milos instructions) <br> [x] get Claude pro |
| June 23 | [x] implement the FareMark paper and reproduce the results <br> [x] run all basic experiments from the paper and obtain proof that code is good <br> [x] deep dive into code - documentation and compare with algorithm in paper to make sure everything is correct <br> [x] short presentation for JSM  to prove everything is working <br> [x] deep dive into the paper and code |
| July 2 | [x] finish up code <br> [x] play around with settings and figure out new attacks <br> [x] create plots and graphs for next JSM presentation |
| July 7 | [x] send a follow up email to authors <br> [x] cleanup codebase (including documentations) and results - get clean results and only keep necessary ones in a summary <br> [] explore better attacks <br> [] explore theoretical approach |
| July 14 | [] broad submarine attacks |
| July 21 | [] fix all the code issues <br> [] review all code and be up to date <br> [] run baseline attack experiments <br> [] analyse results and figure out next steps and feasibility of project |
| July 28 | [] |
| August 4 | [] start writing report ? |
| August 11 | [] |
| August 18 | [] |


---
### submissions
| Date | Tasks |
|------|-------|
| August 21 | - last day |


### NOTES/questions
June9:
- graph colouring - number of nodes and number of colours = number of unique classes needed for watermarking
- federated learning but no data privacy ?
- goal: attack method that utilizes the least amount of resources (eg. only train on the trigger class) to be a free-rider and then test the detection method on it that based on watermarking in outer layer. no matter the data boundary, the free-rider will be detected. => watermaking/fingerprinting on output layer is impossible (with certain conditions).
- collusion attack ? every random amout of rounds ? neighbouring clients ?
- train-then-attack on varying random rounds instead of just beginning ? only trigger sample + certain from others ? mixed attacks etc. predict when to free-ride ?
- penalty for free-riders ? how to mitigate them ?
- facking fairness paper ? optimal transport ?
- attack: threshold is averaged

June16:
- NOTE FOR DATA PARTITIONING - IID for controlled -> QUESTION: more clients than classes table IX
- NOTE FOR calibrating n threshold + sliding window ?
- TODO: test on non-iid
- NOTE: more graphs and plots for the results and experiments
- reputation system for claculting threshold ? dynamic for rounds
- IDEA: plotting attack effort vs detection accuracy - worth the effort or not. how to measure this?
    - num samples, compute it takes

June23:
- QUESTION: what is the clear goal - proving paper has weakness/limitation ? or that paper's definition of effort vs. free-riding is too low for worth ? the paper seems to assume a lot of things - brushing the rest aside as too high effort to be worth free riding - can we challenge that ?
    - ANSWER: yes, we start with challenging paper's assumptions by building an attack that is low effort and but can break through the watermarking detection. explore different attacks and measure the effort vs. detection accuracy. the global goal is to prove thoretically that it is impossible to have watermarking robust in the output layer. 
- QUESTION: non-iid tested in paper ? + data partition the paper does for when too many clients vs clasess - they claim it still works fine
    - non-iid would be weak - maybe even with their current weaker free-rider attacks
- QUESTION: what metric to measure success is prefered? 
    - BER works
- QUESTION: exploring multiple attacks? exploring reputation system? exploring collusion?
    - explore a few, find the lowest effort ones that work
- QUESTION: are we following the paper's assumption for data partitioning? or real FL for data privacy?
    - server has everything but not clients. keep this assumption for now

attack ideas:
- collusion
- threshold weakness - circulatrity on "trusted"
- memory-enhanced beta - global ??? no explanations on tuning
- non-iid missing 
- data paritioning weakness
- attack timing - train-then-attack and trigger-sample-only
    - detection functino, watermark hgih - vs num samples used (num queries)

July2:
- for every plot from now on add standard deviation based on the seeds
- only do one axis plots from now on, no dual y-axis plots
- note for non-iid: interesting. its not an attack but it shows weakness from the paper that we can build and improve on. free rider power doesn't depend on non-iid but it also shows that free-rider doesn't break down during non iid
- add a plot to show the difference between honest and free-rider BER - to show the squeezing effect
- note for cheap evasion attack: check how the extra common samples were sampled, and how the free-riders are detected. plot the effort vs detection accuracy and just re-run good experiemnts for this attack
- check other papers like FedIPR and see if they also only use previous model and gaussian noise attacks. why does faremark only use these 2 attacks - it feels weak
- metrics for "cheap": 
    - compute cycles
    - training time
    - CPU time
    - number of samples used
    - number of forward passes
- new attack ideas:
    - momentum: initially do more work and then benefit afterwards
    - flappy-bird/submarine attack: for any type of free-riding attack (free-ride with previous model, gaussian etc.), train in the beginning, just enough to pass the threshold, then stop training for a number of rounds - use the approximation of the threshold (by using the formula) and your own BER to predict where the threshold is, and then continue training when needed. any way to stay right under the threshold, only training when needed to stay under and then free-ride. make sure to use the standard deviation for this! important to check the recovery time (slower recovery the less you need to train) and how fast the degradation (faster - better ?). compare the compute for honest client, free-rider, and this attack. try this with both iid and non-iid for reference.
    - check the memory enhanced, if its done by the client and if that can be exploited. free-riders can take advantage and just never take the global?


July7:
P:
- S1:
    - submarine attack: warmup by training rounds on full shard until under threshold, then coast until needed to tap again
    - same default experimental setup for now
- N1: no results yet - still running some experiments
    - training just the trigger sample did not work
    - trying out different warmup rounds, dynamic warmup rounds etc.
    - starting with more effort to see if it can be reduced later
- Q1: output layer embedding - alone does not work? prove that you need another detection with it like stale etc
- Q2: timeline plan ?

- collusion - estimate by avg
- start fixing the threshold when the free-rider starts free-riding (assumption - no one will free ride before first 10)
- estimated threshold minus some delta to be safer -> how close you are to the surface (defense)

July9:
- TODO
    - update STATUS.MD
    - finish the slides and plots for the meeting
    - checkup on the 3 seeds run and the final plotting
    - cleanup the results dir
    - cleanup and document the codebase
- NOTES
    - try to find a way to estimate the threshold better - should stay under the actual (maybe adjust the delta?)
    - shallow vs coasting

- try with if free-rider knowns the threshold
- how to find the delta
- experiment on findind the threshold
- how to trick the threshold
- optimal delta and attack

- test if need samples from non-trigger
- effort bar plot - do not use dive cost (+ what is it exacatly - why does it vary) -> 250 batches is a lot as well
- write the algorithm - pseudo - schematic diagram => next meeting have a storyline with algorithm and research question
- double check the block - do exact and check that block code is right -> doesnt make sense that it keeps dipping down instead of submarine

July16
- current issues
    - estimation of threshold: 
        - how to estimate the threshold better - should stay under the actual (maybe adjust the delta?)
        - how to find the delta
        - witholding too little for self-probing that its not accurate enough to estimate the threshold
        - thinks it's safe but its not - not tapping when it should - TO FIX
        - using the estimated threshold but using probe BER (not enough) to check if it's under and its not - TO FIX
- next steps
    - hard/easy position debate
    - iid vs non-iid
- TODO
    - use class id not position
    - CHECK THE THRESHOLD: double check that 3 standard deviation - should be 99% of all classes, clearly not shwoing that in the plot. the 0.1 line std dev should be much higher since its 3x
    - check why there is a flat line -> reason for the BER to be flat? plot the function loss (if that is moving but the BER stays flat its a bit suspicious? unless i can proove that it has just reached the best it could and converged). look at the dynamics of a single round. why does each class do something different (harder boundary for some classes? per class accuracy - see losses - see if some classes are harder? plot the model accuracy and loss side by side). check free-rider accuracy too.
    - check if using CIFAR-10 (only 10 classes?) -> using CIFAR-100 but with 10 clients so 10 trigger classes
    - focus on iid for now, later collusion and non-iid could make sense if the effort is worth it

repeated prisoner dilemna

july20
- threshold 
    - the way its calculated now + hpow low it is
- class difficulty
- better watermark embedding with less data ?
- more free-riders falls under threshold with less good global accuracy
- the way they calculate FPR?

---
---
## resources
#### my code
- [forked-repository](https://github.com/zu-greta/decentralizepy)
    - FareMark reproduction code and my own implementation of the paper can be found under the `faremark_greta` folder. The code is structured in stages and each stage has a correctness gate to ensure that the foundation is sound before building on it. The code is modular and can be ported into decentralizepy later if we need genuinely distributed runs.
- [original-code](https://github.com/sacs-epfl/decentralizepy) - original code from the SaCS lab for decentralized federated learning


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
---