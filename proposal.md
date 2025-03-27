---
layout: page
title: Project Proposal

---

# Parallel Graph Coloring Algorithms

## Summary

We are going to implement two related parallel graph coloring algorithms on the GHC and PSC machines on CPUs using OpenMP.

## Background

Graph coloring is the assigning of labels – often represented as colors – to a graph. A graph is a network structure where individual nodes (also known as vertices) are connected by edges. In order for a graph coloring to be valid, no two nodes with the same color can be connected by an edge. Trivially, one could assign a different label to each node and have a valid graph coloring. But in practice, the goal is generally to use a small set of labels to color the graph. The smallest possible number of colors a graph can be colored by is called its chromatic number.

Graph coloring started out as a problem in pure mathematics. But since then, various applications have been found. The common application area is scheduling. In scheduling, a number of different jobs must be scheduled into time slots. But some jobs cannot be scheduled in the same time slot, often because of some shared resource. We can think of these jobs as a graph. Each job is a node. Jobs that cannot occupy the same time slot are connected by an edge. The labels are the time slots. This is often used in parallel computation where certain jobs cannot or should not be scheduled at the same time.

In graph coloring, there are two conflicting goals. First, one wants to use few colors, maybe even the fewest possible. On the other hand, efficiency is an important goal. If one is scheduling parallel computations, then one does not want to waste too much time simply computing the graph and it may make sense to produce a suboptimal coloring if the time it would take to generate a perfect coloring would be much greater than the time saved. The balance between these two goals depends on the precise application.

The algorithms we are implementing are heuristic based algorithms that, roughly speaking, iterate through nodes and assign them label ids that are the smallest possible given their neighbors. Our algorithm is meant to be used in scheduling situations when each node represents a small amount of computation, and therefore we care more about the efficiency of producing the coloring than using a small set of colors. In particular, our algorithm first generates a pseudo-coloring that has a small number of conflicts and then resolves the small set of conflicts sequentially.

See the resources section for the pseudocode as presented by [1].

## The Challenge

There are a few different challenges in parallel graph coloring. The first major challenges are distributing work and communication or memory sharing. To color a node, one must know the colors of its neighbors. If any of those neighbors are on another node, this requires communicating or sharing memory with those nodes. This is complicated by the fact that nodes are getting written to in the process of coloring, invalidating other processors’ copies.

Another problem is the inherently sequential nature of the problem. In order to determine what label to give a node, one must know the labels of the neighbors. But if those neighbors have not yet been colored, what is one supposed to do? We can use techniques like locks, communication, and other techniques to ensure that one neighbor is colored before the other but this introduces its own issues, as alluded to in the previous paragraph. This means we cannot simply assign every single node a label in one time step in isolation from each other.

Another challenge is graph representation. Graphs can be represented in a variety of ways, They can be represented as lists of nodes and edges. Alternatively, they can be represented with pointers in each node that reference neighbors. We have to pick a representation that is space efficient and makes computation easy.

## Resources

We will be following the synchronous and asynchronous versions of the algorithms provided in [1].

We will create our own code from scratch and will target the GHC machines. 

We will also use real-world graph datasets to test the algorithms. We are yet to pick particular datasets.

## Goals and Deliverables

### If the work goes slowly

- Create a baseline of a greedy sequential algorithm which iterates through each node and assigns it the label with the lowest id that is not shared by any already colored neighbors.
- Implement the synchronous version of algorithm 1 from [1] in OpenMP.
- Test cases utilizing a few different graphs.We plan to do the testing on both GHC and PSC.

### Plan to Achieve

- Implementation of both the synchronous and asynchronous versions of algorithm 1 and 2 from [1] in OpenMP.
- Test cases for the algorithm utilizing a variety of graphs, including generated ones and ones from real world datasets. We want to see how algorithm performance relates to graph structure. We plan to do the testing on both GHC and PSC.

### Hope to Achieve

- A modification of algorithm 2 where step 2 of the pseudocode as presented in [1] (the refining of the pseudo coloring) is repeated several times. We will repeat either a number of times given by the user as an argument or until convergence.
- A modification of the algorithms where some nodes in the input graph are already colored and are not permitted to be recolored.

## Platform Choice

We plan to use OpenMP on the GHC and PSC machines. A common application of graph coloring is scheduling tasks on a parallel machine. We are aiming to create a graph coloring algorithm that could be used by someone working on a single node machine. While message passing is good for clusters, a shared address model is better for a single node. 

Using message passing, we would need to replicate a lot of data. Threads need to be able to access nodes being processed by other threads because those nodes could be neighbors. This would mean that every thread would need to maintain copies of all the neighbors of the nodes it is assigned.

## Schedule

- First week (March 23 to March 29): Implement a working sequential graph coloring algorithm. Create test cases

- Second week (March 30 to April 5): Implement algorithm 1

- Third week (April 6 to April 12): implement algorithm 2

- Fourth week (April 13 to April 19) : Iteratively test and refine the implementations of algorithms 1 and 2 to improve efficiency without straying from the algorithms provided in [1]

- Fifth week (April 20th to April 26th): Write a report and create a poster. If possible, fulfill stretch goals (see “hope to achieve” goals)

- Sixth Week: Poster presentation and Final Report Due


## References

[1] Gebremedhin, A.H. and Manne, F., 2000. Scalable parallel graph coloring algorithms. *Concurrency: Practice and Experience*, *12*(12), pp.1131-1146.