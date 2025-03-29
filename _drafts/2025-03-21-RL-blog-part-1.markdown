---
layout: post
title:  "I tried RL to solve a scheduling problem."
date:   2025-03-21 21:57:00 -0400
categories: reinforcement_learning
---
Over the last few weeks, I tried reinforcement learning to solve a scheduling
prblem.

This was my first "RL project", so there were many trial and error throughout
the process. There are still many areas of improvement I want to make, but I
thought it's worth sharing my experiences so far.

Note that I won't go over much details on each of the algorithms I used - I'll
mostly focus on empirical observations I found when I applied them.

## Problem Statement
The project idea came out when I was talking with my girlfriend about her work.
She said that there are companies that own multiple clinics and help patients
get outpatient care. The company needs to hire nurses to handle these patients.
There are many interesting operation research-esque optimization problems you
can formulate, but I simplified the problem into the following:

- There are $C$ number of clinics, where each clinic can handle $n_{c_i}$ number
  of patients, where $i \in \[C\]$.

- The time it takes to go from one clinic to another will also be given as a
  matrix format $M$, where $M_{ij}$ represents the time it takes to go from
  $c_i$ clinic to $c_j$ clinic.

- There are $U$ number of nurses. Nurses can either treat a particular patient
  or drive to a particular clinic.

- There are $P$ number of patients. Each patient has different treatment time,
  which we call $p_i$.

- However, each patient doesn't need nurses' care for the entire treatment time.
  They only need their care for the first and last 15 minutes of their treatment
  (so assume that the treatment time takes at least 30 minutes).

The problem initially looked a lot like a [constraint
programming](https://en.wikipedia.org/wiki/Constraint_programming) problem, but
it wasn't initially obvious how the problem could be translated into one (e.g.
making sure that the nurse has moved from one clinic to another). I'd love to
hear thoughts from others whether this is feasible or not!

However, the entire setup seemed easy to simulate, so I looked into
reinforcement learning.

## Simulation

First, I built a simultion environment to start. I used
[Gymnasium](https://github.com/Farama-Foundation/Gymnasium) framework, which
gives nice utility functions to convert the observation/state of the environment
into commonly used formats like flattened numpy array.

One thing I would've done differetly if I rebiuld the simulation is that I would
keep the state of the environment as "raw" as possible and leave all the
pre-processing part to the later parts. As you'll see in the later discussions,
the types of pre-processing of the environment before you inputted into the
agent/model varied quite a bit, and a premature pre-processing on the
environment had to be removed later.

To be concrete, I created an environment where during each turn, a nurse can
take one of three actions: treat one of the patients if they need care, or move to
a different clinic, or take no action. Since we care about a global schedule
across nurses, I framed the question as a single-agent problem where it's
responsible for scheduling all the nurses and patients.

## Brief Intro to Reinforcement Learning

If you're already familiar with reinforcement learning, feel free to skip this
page. I'll give a very brief recap on RL core concepts. If you'd like to learn
more, I found the classic book [Reinforcement Learning: An
Introduction](http://incompleteideas.net/book/the-book.html) very useful.

Assume your environment is in a markov decision process (MDP), where the next
state and the reward you'll immediately receive is only determined by the
current state and the action (so previous state doesn't affect the next
state/reward).

Our goal is to learn a function called "policy", $\pi: S \rightarrow A$, which
will take the state of the environment $s \in S$ as an input and outputs the
best action $a \in A$ to take in such state. "Best" means "highest total sum of
rewards", including the discount factor if applies. We'll call such sum of
rewards returns.

One way to learn $\pi$ is to learn what we call a Bellman equation. Bellman
equation represents an expected return based on the current state, and sometimes
with action, and the policy. State-based Bellman equation is:

$$ V^{\pi}(s) = \max_{a}E_{s'}[R(s, a) + \gamma V^{\pi}(s')] $$

The state-action-based Bellman equation is:

$$ Q^{\pi}(s, a) = E_{s'}[R(s, a) + \gamma \max_{a}Q^{\pi}(s', a)] $$

If you assume that you found an optimal Bellman equation $V^{\*}$ or $Q^{\*}$,
then the policy is straightforward: for each state you visit, compute the
Bellman equation for every actions, and choose the one that gives the maximum
expected return. In reality, we don't know such function in advance, but there's
some [cool theory](https://en.wikipedia.org/wiki/Banach_fixed-point_theorem)
that tells us that it's possible to iteratively improve the policy by updating
its value function.

Another way to learn $\pi$ is to approximate $\pi$ directly - this is called
policy gradient method, which is to define an approximate function for $\pi$ and
continuously improve it. Examples algorithms include REINFORCE, Actor-Critic,
etc.

## Reward Function
Setting up an appropriate reward function turned out to be one of the majors
tweaks to help agents learn. One interesting observation I found was the
relationship between incentivization and penalization ("carrot and stick").
Initially, whenever the environment terminated because the patient didnt' get
the treatment at the last 15 minutes early enough, I penalized the agent with a
negative reward. If the agent sucessfully treats all the patients, I gave a
reward that's monotonically decreasing based on the completion time.

However, this led to an agent to not take any actions because the environment
had a lot of sequence of actions that led to penalizations, so an inaction was
technically giving more rewards than an action (even though I rewarded the agent
if it treated all the nurses). Removing the penalization (so 0 reward will be
given when it terminates early) changed the behavior of the agent to more
eagerly explore the environment to find a better reward.

I believe this is a type of [reward
hacking](https://en.wikipedia.org/wiki/Reward_hacking). It was useful to have a
rough sense of what the "optimal" value function would look like to debug the
agent behavior. However, this is probably a lot harder if the reward function
gets complicated. Ideally, it'd be nice to not do too much reward shaping, but
there are known difficulties to this, too.

## Tabular Learning

After building the simulation, I first tried a very basic RL algorithms as a
sanity check - tabular RL. This unsurprisingly didn't learn well when the number
of patients grew to 3 (?).

Two methods I tried were SARSA and Q-learning, which aim to learn the
state-action value function $Q$. Because the update of both SARSA and Q-learning
is based on the reward the agent recevied, and the agent only received a
positive reward at the end, the value function for early states stayed at the
initial value for a long time.

Even though it was a good exercise to get hands a bit dirty with tabular
learning, it was soon obvious that a more complex tool was necessary.

## DQN

After reading the [classic](https://arxiv.org/abs/1312.5602) paper, I started
implementing a DQN algorithm. There were lots of practical knowledge I gained
throughout the implementation process, but I'll focus on three: basic neural net
training, experience replay, and masking invalid moves.

### Basic Neural Net Training

DQN approximates a Q function via a nerual network. A common network for DQN
will take an environment state as an input and output the value function of all
the actions in the action space because evaluating Q function requires taking
the max among all the Q functions.

The main change I had to make to make the neural network work is to normalize my
inputs and outputs to avoid gradient explosion. I added a few functions inside
the custom gym environment to normalize the environment state mostly based on
heuristics.

However, normalizing the inputs wasn't enough - without normalizing the
network's outputs, I still had a gradient explosion frequently. Normalizing the
output is trickier, because based on how you define your reward function, the
range of the reward function might be hard to bound.

In this case, once I changed my reward function to only give positive reward at
the successful termination (and no negative rewards), and the positive reward
was given as one over the number of iteration taken to finish, the value was
bounded to $[0, 1)$. Thus, I added a tanh function with an offset to bound the
network output to such range. Adding this "post-processing" of the raw outputs
removed any gradient explosions.

<!--
#### Sidenote: Framing neural network training as an engineering problem
A lot of the neural network materials initially felt overwhelming - there are
lots of terminologies, 
-->

### (Prioritized) Experience Replay

To create a batch of samples to learn from, DQN algorithm keeps a history of
recent state transitions and updates and samples from such buffer to do a
gradient descent. A literature calls this an experience replay.

The most straightforward implementation of this is to have a deque to push new
state transitions and remove the old ones, and sample randomly from this deque.
However, this implementation has a downside: it's probably not true that all
the observations are equally important. Then, how can we define which samples
are more important than others, and what should we do with that?

This is where [prioritized experience replay](https://arxiv.org/abs/1511.05952)
comes in. Prioritized experience replay defines the importance of the sample
based on the l2 loss it had when it was chosen as part of the batch previously.
Then, the replay buffer does a weighted sampling based on the weights (There are
more details like applying an importance sampling to the batch, but I won't go
over the details).

However, creating a reasonable data structure to build a prioritized experience
replay turned out to be pretty tricky - sampling a batch shouldn't be dependent
on the length of the buffer, otherwise it'll be too slow. I ended up
implementing a ranked-based approach based on the paper, which keeps the
elements in a roughly sorted manner, divide the buffer into buckets such that
each bucket has the same cumulative weight, and samples from each bucket.

Sadly, the training stil took a nontrivial amount of slowdown, and the
performance improvement wasn't hugely obvious with this change. However, it was
one of the two features that added the most value in the [rainbow
paper](https://arxiv.org/abs/1710.02298), so it may be worth revisiting this to
find potential bugs.

### Masking Invalid Moves

The most significant improvement I observed actually came from removing invalid
moves from the consideration when updating the Q function. Because the
environment by nature has many invalid moves in each state transition, without
forcing the network to ignore invalid moves, the search sapce had too much
garbage for the agent to have a useful information.

To enforce this, I updated the environment to give a list of valid actions for
each state transition and saved such mask in the experience replay as well.
Then, I also masked invalid moves when updating the Q function. This led the
agent to explore and learn more efficiently.

### Challenges

Even with all the optimizations above, DQN setup didn't pan out as much as I
wished. One of the reasons is the exploration - even though the agent couldn't
make an obviously invalid move (e.g. the nurse can't travel while they are
treating a patient), they could still make moves that will eventually lead to
patients not getting treated on time (e.g. moving to a different clinic when the
patient at the current clinic needs treatment very soon).

Even though I believe the model eventually learned these moves, because the
agent explored using $\epsilon$-greedy exploration, the exploration usually led
to a dead end often. I think there's a better way to explore the space than just
choosing a random action, but I haven't explored this yet.

Also, even though there are many great RL frameworks, I found it difficult to
use them "off the shelf" - e.g. the framework had a DQN algorithm, but I wanted
to make changes on the neural network to include invalid actions. I probably
haven't explored the tools much yet, so if you think there's a tool I could use,
please let me know!

## Policy Gradient, and Attention

After numerous unsuccessful attempts to make DQN setup work, I went back to
research papers to see whether there are any previous works done to solve
problems like this. Luckily, there were a few promising papers related to
solving combinatorial optimization problems (e.g. traveling salesman problem
(TSP), binpacking, etc.) using reinforcement learning!

While reading the [survey paper](https://arxiv.org/abs/2003.03600), I found out
that a lot of the successful approaches had two common characteristics: they
often used a policy gradient approach, and they used a more sophisticated neural
network architecture than just an MLP. I'll focus on addressing my experiences
while working with these two techniques.

Before going into the two techniques, I'd like to say a big thank you to the
authors of ["Attention, Learn to Solve Routing
Problems!"](https://arxiv.org/abs/1803.08475). The paper is a great read, and a
lot of the ideas below are inspired from it.


### Policy Gradient

An immediate benefit I saw from the policy gradient approach compared to DQN was
the stability of the agent over the training. As I addressed above, because of
$\epsilon$-greedy exploration, the DQN agent's total reward was pretty flaky if
their random action led to an invalid move. However, because the neural network
to approximate the policy was fully responsible for the exploration, the policy
gradient-based agent didn't suffer from a heuristic-based exploration that DQN
faced.

However, this didn't mean that the algorithm didn't have its fault. One major
problem I saw was that it was very easy for the agent to converge to a locally
optimal solution. Once it reached to the local optima, the gradient quickly went
to zero and didn't explore.

Why would this happen? One hypothesis is that the gradient is heavily
incentivized to move to the first successful solution. In the current setting,
because the agent will receive zero reward if they fail to treat all the
patients, any failed attempt will give zero gradient. When it's rare for the
agent to get any reward, the policy will be updated to be closer to the first
policy that finds _a_ solution.

I have not tried it yet, but one way to prevent this is to [give more rich
environments](https://arxiv.org/pdf/1707.02286) to prevent "overfitting". Even
if we're interested in finding the optimal solution for a particular clinic
setup, by giving a similar environment during training (e.g. give patients of
different treatment time, different distance between clinics, etc.), the agent
is less likely to overfit.

### Attention

One interesting aspect of the problem is that there's a relationship between the
environment characteristics and the set of actions the agent can take - the set
of actions is defined by the number of patients and clinics the environment has.
This setup is also similar for TSP - an "action" the salesman can take in a
given state is based on the the set of nodes provided by the graph.

One striking observation is that there's a parallel structure between choosing
the next best action in our environment and choosing the next best token to
choose based on the current sequence of tokens, which as we all know the
attention architecture is great at! To be more concrete, we can create a network
where:

1. Encode the environment by embedding the state of each entities (nurses,
   patients, clinics) and apply self-attention to each entity. Unlike the
   sequence model, we don't need to include positional encoding.
2. Decode the state by applying attention from the encoded state to the
   embeddings of "actionable" entities (patients and clinics) to choose the best
   action to take.

["Attention, Learn to Solve Routing
Problems!"](https://arxiv.org/abs/1803.08475) uses such idea to solve TSP, and
the model seems to behave pretty well in our setup, too.

## Conclusion

Even though the project still has a lot more improvements to be made, I learned
a lot from this journey. One thing that particularly excites me the most is that
there are lots of opportunities that can make tooling better to make RL research
better - simulating many environments in parallel, finding the right amount of
abstractions to easily "plug and play", etc. 

I'm thinking of focusing on engineering to make the research environment better
and revisit the problem. If you have any suggestions on how to make RL research
experience better, I'd be happy to hear your thoughts.
