# Documentation Index

Read these documents in order when implementing the first prototype.

1. [Project overview](project_overview.md)
2. [Lessons from GenFEC](lessons_from_genfec.md)
3. [System architecture](architecture.md)
4. [LLaDA decoding design](llada_decoding_design.md)
5. [Implementation plan](implementation_plan.md)
6. [Experiment plan](experiment_plan.md)
7. [Development notes](development_notes.md)
8. [Reference sources](reference_sources.md)
9. [Server SSH workflow](server_ssh_workflow.md)
10. [Agent handoff](AGENT_HANDOFF.md)

The most implementation-critical file is [LLaDA decoding design](llada_decoding_design.md). It defines the constrained denoising loop: masked target tensor, fixed-token restoration, hash-bucket logit masking, low-confidence commit policy, and diagnostics.
