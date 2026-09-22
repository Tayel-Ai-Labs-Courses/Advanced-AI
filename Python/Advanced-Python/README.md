# Track 2 — Advanced Python

Track 1 taught you to make the computer do what you want. Track 2 is about
making it do so efficiently, and being able to explain why your solution is the
right one — which is what interviews and code reviews actually test.

## Lessons

| # | Lesson | You will be able to |
|---|---|---|
| 01 | [Advanced Functions](lessons/01-advanced-functions.md) | Use closures, decorators, `functools` |
| 02 | [Iterators and Generators](lessons/02-iterators-and-generators.md) | Process data larger than memory |
| 03 | [Complexity and Big-O](lessons/03-complexity-and-big-o.md) | Say why code is slow, before running it |
| 04 | [Searching](lessons/04-searching.md) | Binary search, and hashing as search |
| 05 | [Sorting](lessons/05-sorting.md) | Know what `sorted()` does and when to help it |
| 06 | [Core Data Structures](lessons/06-core-data-structures.md) | Stacks, queues, linked lists, hash tables |
| 07 | [Trees and Heaps](lessons/07-trees-and-heaps.md) | BSTs, traversals, priority queues |
| 08 | [Graphs](lessons/08-graphs.md) | BFS, DFS, shortest paths |
| 09 | [Structures for AI Engineers](lessons/09-structures-for-ai.md) | Vector search, tries, caches, batching |
| 10 | [Concurrency and Performance](lessons/10-concurrency-and-performance.md) | Profile first, then parallelise correctly |
| 11 | [Typing, Testing, Packaging](lessons/11-typing-testing-packaging.md) | Ship code other people can rely on |

## Then

- [`Project-2/`](Project-2/) — build a semantic search engine, from scratch

---

## How this track is different

Track 1 asked "does it work?". Track 2 asks three more questions:

1. **How does it scale?** Working on 100 rows and working on 100 million are
   different claims.
2. **What does it cost?** Time and memory, stated in Big-O, not vibes.
3. **Can someone else maintain it?** Types, tests, and a name that explains
   itself.

Implement every data structure in this track by hand once. You will then use
the library version for the rest of your life — but you will know what it is
doing, which is the difference between choosing a tool and guessing at one.
