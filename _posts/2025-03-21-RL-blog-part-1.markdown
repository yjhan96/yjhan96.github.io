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

Our goal is to learn a function called "policy", $\pi: S -> A$, which will take
the state of the environment $s \in S$ as an input and outputs the best action
$a \in A$ to take in such state. "Best" means "highest total sum of rewards",
including the discount factor if applies. We'll call such sum of rewards
returns.

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
negative reward.

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

Two methods I tried were SARSA and Q-learning. Both methods aim to learn the
state-action value function $Q$. 

## DQN

## Policy Gradient

## Conclusion
