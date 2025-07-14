# Cloud-Scale External Sorting Research Proposal

## Background

External sorting (sorting data sets that no longer fit in memory) has been a long-standing research problem for the database and systems communities. In cloud-based data processing, data now lives primarily in disaggregated object storage; when working sets outgrow memory, most major operations turn into a shuffling problem on top of that storage layer. Prior studies explored optimizing out-of-memory operations by spilling to fast local SSDs, but such hardware is rarely available on public clouds, with SSDs not having been updated since 2018.

As a result, performance of an external sort operation depends more on topology-level choices: the fan-out and fan-in of the shuffle layer, the network bandwidth, the run-generation and merge algorithms, and the degree of data or worker skew that can drive an algorithm from average- to worst-case complexity. Each parameter affects not only end-to-end latency but also resource consumption and, ultimately, dollar cost, since every shuffle request incurs compute charges (servers are billed for their uptime) and per-operation fees to the object store. Understanding these effects requires a cloud-centric perspective on external sorting.

## Project Goal

Our research goal is to revisit cloud-scale sorting starting with a plug-and-play simulator that predicts time-and-cost trade-offs for multiple algorithms under diverse shuffle configurations, and then validate those models experimentally.

## Experimental Requirements

To achieve reproducible, large-scale validation, we require:

- **User-defined network topologies**: create and test custom cluster layouts.  
- **Controlled link shaping**: precisely configure latency and bandwidth between nodes.  
- **Bare-metal disk access**: enable fine-grained I/O tracing at the block-request level.


## Impact

The insights and validated models from this project will:

- Illuminate how topology, bandwidth, and data skew co-determine external-sorting performance and cost in cloud environments.  
- Guide architects of next-generation cloud data-paths toward designs optimized for large-scale out-of-memory operations.

---

**Project Title:** Simulating Time-and-Cost Trade-offs in Cloud-Scale External Sorting
