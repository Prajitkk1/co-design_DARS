# A Talent-infused Policy-gradient Approach to Efficient Co-Design of Morphology and Task Allocation Behavior of Multi-Robot Systems
[![DOI](https://sandbox.zenodo.org/badge/693318035.svg)](https://sandbox.zenodo.org/doi/10.5072/zenodo.100837)

## Co-design
 This paper proposes a computational framework that enables co-optimization of morphology and behavior of individual robots in a swarm to achieve maximum performance from its emergent behavior while also helping explore the benefits of swarm systems. Here, we utilize our previously proposed artificial-life-inspired talent metrics that are physical quantities of interest, reflective of the capabilities of an individual robotic system. Talent metrics represent a compact yet physically interpretable parametric space that connects the behavior space and morphology space. We use this to decompose the morphology-behavior co-optimization into a sequence of talent-behavior optimization problems that can effectively reduce the overall search space (for each individual problem) without negligible compromise in the ability to find optimal solutions. In other words, the decomposition approach presented here is nearly lossless, i.e., a solution that can be found otherwise with a brute-force nested optimization approach to co-design will also exist in the overall search space spanned by our decomposed co-design approach (albeit assuming that each search process is ideal). We also propose a novel talent-infused policy gradient method to concurrently optimize the talents and learn the behavior.
The framework consists of 4 steps: a) Initally Morphology and its dependent talent parameters are derived, b) Based on the talents, we create a Pareto front by solving multi-objective optimization, c) The Talent-infused policy-gradient method is used to train the associated behavior and talents, d) Finalize the morphology.
 [DARS_framework (2).pdf](https://github.com/user-attachments/files/16626957/DARS_framework.2.pdf)

## Multi-robot Task Allocation for Flood Response
In this work, we specifically focus on multi-robot disaster response problem, which we refer to as MRTA-Flood. It consists of $N_{T}$ locations in a flood-affected area waiting for a survival package to be delivered by a team of  $N_{R}$ robots. Here, the mission is to drop survival packages (referred to as the task) to maximum locations before the water level rises significantly, submerging all the locations. 
We assume each location requires just one survival package. The predicted time at which a location $i$ gets completely submerged ($\tau_i$) is considered as the deadline of the task $i$. A task $i$ is considered to be successfully done if it is done before time $\tau_i$. Each robot has maximum package capacity, speed and range decided by the talents. We consider a decentralized asynchronous decision-making scheme. The following assumptions are made: 1) All robots are identical and start/end at the same depot; 2) The location $(x_i,y_i)$ of task-$i$ and its time deadline $\tau_i$ are known to all robots; 3) Each robot can share its state and its world view with other robots; and 4) A linear charging model with a charging time from empty to full range being 50 minutes, the charging happens every time the robot visits the depot. 

### MDP Formulation:

The MRTA-Flood problems involve a set of nodes/vertices ($V$) and a set of edges ($E$) that connect the vertices to each other, which can be represented as a complete graph $\mathcal{G} = (V, E, \Omega)$, where $\Omega$ is a weighted adjacency matrix. Each node represents a task, and each edge connects a pair of nodes. For MRTA with $N_{T}$ tasks, the number of vertices and the number of edges are $N_{T}$ and $N_{T}(N_{T}-1)/2$, respectively. Node $i$ is assigned a 3-dimensional feature vector denoting the task location and time deadline, i.e., $\rho_i=[x_i,y_i,\tau_i]$ where $i \in [1, N_{T}]$. Here, the weighted between two edges $\omega_{ij}$ ($\in \Omega$) can be computed as $`\omega_{ij} = 1 / (1+\sqrt{(x_{i}-x_{j})^{2} + (y_{i}-y_{j})^{2} + (\tau_i - \tau_j)^2})`$, where $`i, j \in [1,N_{T}]`$.
The MDP defined in a decentralized manner for each individual robot (to capture its task selection process). The components of the MDP can be defined as 

State Space ($`\mathcal{S}`$): A robot $r$ at its decision-making instance uses a state $`s\in\mathcal{S}`$, which contains the following information: 1) Task graph $`\mathcal{G}`$, 2) the current mission time $t$, 3) the current location of the robot ($`x^{t}_{r}, y^{t}_{r}`$), 4) remaining ferry-range (battery state) of robot $`r`$ $`\phi^{t}_{r}`$, 5) capacity of robot $r$ $c^{t}_{r}$ , 6) destination of its peers ($`x_{k}, y_{k}, k \in [1, N_{R}], k \neq r`$), 7) the remaining ferry-range of peers $`\phi^{t}_{k}, k \in [1, N_{R}], k \neq r`$, 8) capacity of peers $`c^{t}_{k}, k \in [1, N_{R}], k \neq r`$, 9) next destination time of peers $`t^{next}_{k}, k \in [1, N_{R}], k \neq r`$, and 10) the talents $`\hat{Y}_{\texttt{TL,1}}$ and $\hat{Y}_{\texttt{TL,2}}`$. For the MRTA-Flood problem, we assume that each robot can broadcast its information to its peers without the need for a centralized system for communication, as aligned with modern communication capabilities. 

Action Space ($`\mathcal{A}`$): The set of actions is represented as $`\mathcal{A}`$, where each action $a$ is defined as the index of the selected task, $`\{0,\ldots,N_{T}\}`$ with the index of the depot as $0$. The task $0$ (the depot) can be selected by multiple robots, but the other tasks are allowed to be chosen once if they are active (not completed or missed tasks). 

Reward ($`\mathcal{R}`$): The reward function is defined as
$`10 \times N_{\text{success}}/N_{T}`$, where $`N_{\text{success}}`$
is the number of successfully completed tasks and is calculated at the end of the episode.  

Transition: The transition is an event-based trigger. An event is defined as the condition that a robot reaches its selected task or visits the depot location. Since here we do not consider any uncertainty, the state transition probability is 1.

## To train Co-design:
run training_mrta.py



