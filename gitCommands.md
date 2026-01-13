# Git Command Log

This document tracks the git commands executed during the development process, along with explanations for why they were run.

## Initial Setup & First Commit
**Date**: 2026-01-13
**Feature**: Initial Repo Setup

1.  `git status`
    *   **Reason**: To check the current state of the repository, identify untracked files, and ensure `.gitignore` was working correctly (e.g., ensuring `.env` was not listed).

2.  `git add .`
    *   **Reason**: To stage all current file changes and new files for the next commit. This prepares the files to be saved in the git history.

3.  `git commit -m "Initial commit: Multimodal RAG system with .gitignore"`
    *   **Reason**: To create a saved snapshot (commit) of the staged changes with a descriptive message explaining what was done.

4.  `git push`
    *   **Reason**: To upload the local commits to the remote repository (GitHub) on the `main` branch, syncing the local work with the cloud.

## Phase 3: Agentic Workflow Core
**Date**: 2026-01-13
**Feature**: Branch Creation

5.  `git checkout -b phase-3-agentic-workflow-core`
6.  `git add .`
    *   **Reason**: To stage all changes made during Phase 3 implementation (new files, modified readme, etc.).

7.  `git commit -m "Feature: Phase 3 - Agentic Workflow Core (Router, Grader, Web Search)"`
    *   **Reason**: To save the Phase 3 changes to the `phase-3-agentic-workflow-core` branch.

8.  `git push --set-upstream origin phase-3-agentic-workflow-core`
    *   **Reason**: To backup the feature branch to GitHub.

9.  `git checkout main`
    *   **Reason**: To switch back to the main branch to prepare for merging. (Note: Had to remove `index.lock` and kill processes blocking the checkout).

10. `git merge phase-3-agentic-workflow-core`
    *   **Reason**: To integrate the tested Phase 3 features into the stable `main` codebase.

11. `git push`

## Phase 4: Control Loop & Orchestration
**Date**: 2026-01-13
**Feature**: Graph Control Flow

12. `git checkout -b phase-4-control-loop`
    *   **Reason**: To isolate Phase 4 changes (LangGraph integration) from the main branch.

13. `git add .`
    *   **Reason**: To stage the new Graph implementation, agents, and test scripts.

14. `git commit -m "Feature: Phase 4 - Control Loop & Orchestration (LangGraph)"`
    *   **Reason**: To save the Phase 4 changes.

15. `git push --set-upstream origin phase-4-control-loop`
    *   **Reason**: To backup the Phase 4 branch to GitHub.

16. `git checkout main`
    *   **Reason**: Switching to main for merge. *Note: Terminated running python processes (Streamlit) to release file locks on `chroma.sqlite3`.*

17. `git merge phase-4-control-loop`
    *   **Reason**: To integrate the Agentic Graph Control Loop into the main branch.

18. `git push`

## Phase 5: RAG Security Architecture
**Date**: 2026-01-13
**Feature**: Security Guards & Robustness

19. `git checkout -b phase-5-security`
    *   **Reason**: To isolate security implementations (Guardrails, Policy) from the main branch.
