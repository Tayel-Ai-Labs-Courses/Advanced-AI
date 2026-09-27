# Lesson 07 — Presenting

**Goal:** stand in front of people and get a decision, without a demo that
breaks or a slide nobody can read.

## What you will learn

- The five-slide deck
- Slide rules that survive a real room
- Demos, and how they fail
- Questions, including the hostile one

---

## Five slides

For a fifteen-minute decision meeting:

```text
1. THE ASK          what you want them to decide, and by when
2. THE EVIDENCE     three numbers, one chart
3. THE COST         money, people, time, and what you are giving up
4. THE RISK         the biggest one, and what you will do about it
5. THE DECISION     the options, with your recommendation marked
```

That is the whole deck. Everything else — method, architecture, the experiments
that did not work — goes in an appendix that you will probably not open, and
that exists so you can answer a question by jumping to it.

**Slide 1 is the ask, not the agenda.** A meeting that opens with "today I'll
walk you through our approach" has already spent its best minute. Open with
*"I'm asking for approval to start Monday calls on 6 October; it is worth about
41,700 EGP a month."*

---

## Slide rules

| Rule | Why |
|---|---|
| **One idea per slide** | Two ideas means nobody remembers either |
| **The title is the finding** | Same rule as chart titles (lesson 04) |
| **Six lines maximum** | More and they read instead of listening |
| **No slide you have to apologise for** | "I know this is busy" means delete it |
| **Numbers in the title** | "59 more churners a month" beats "Results" |
| **Readable from the back** | 24pt minimum. Test it standing up |
| **No animations** | They fail on someone else's machine |
| **Dark text on light** | Projectors wash out everything else |

And the rule that saves the most meetings: **do not read the slides.** The
audience reads faster than you speak. The slide is the anchor; you are the
explanation.

---

## Demos

A live demo is the most persuasive thing you can do and the most likely thing
to fail. The failure modes are always the same:

| Failure | Fix |
|---|---|
| The wifi | Run locally. Always |
| A slow model | Pre-warm it before you walk in |
| An API rate limit or an expired key | Record a video of the working demo as a backup |
| An empty database | Seed it beforehand, with data that tells the story |
| An input that breaks it | Rehearse the exact inputs. Write them on a card |
| Someone asks "what if I type this?" | Say "let me show you afterwards" — do not improvise |
| The screen resolution | Test on the actual projector |

**Always have the video.** Ninety seconds, recorded, working. When the live
version fails you say "let me show you the recording" and lose ten seconds
instead of the room.

And rehearse the demo **standing, out loud, once**. Silent rehearsal at your
desk finds none of the real problems.

---

## Questions

Four kinds, four responses:

**The clarifying question** — they did not follow. Answer it plainly, and note
that your explanation needs work.

**The technical question** — a peer testing the method. Answer precisely. If
the answer is in the appendix, go there; that is what it is for.

**The hostile question** — they do not want this to happen. The answer is
never to argue the person; it is to name the disagreement.

> *"You're saying the model will save 41,700 a month. I don't believe it."*
>
> "The number rests on a 30% offer-acceptance rate, which is an assumption, not
> a measurement. At 15% it is 4,000 a month and the project is marginal. That is
> why the proposal is a four-week test with a holdout — after which we will know
> the real rate rather than arguing about it."

You did not defend the number. You **named the assumption, gave the downside,
and turned the disagreement into a measurement**. That is the move, and it works
because it is honest.

**The question you cannot answer** — say so. *"I don't know. I'll find out and
send it by Thursday."* Then send it by Thursday. Inventing an answer in a
meeting is the fastest way to lose a room that was on your side.

---

## Before the meeting

- [ ] You know who **decides**, and what they need to decide
- [ ] The ask is on slide 1, with a date
- [ ] Every number has its interval, and you know its weakest assumption
- [ ] You have pre-briefed the decision-maker, **if the answer might be no** —
      nobody likes being surprised in front of others
- [ ] The demo runs locally, and the video exists
- [ ] The appendix has the three questions you expect
- [ ] You have rehearsed standing up, out loud, once
- [ ] You know what you will do if they say no

The pre-brief is the highest-leverage item. A decision meeting is a bad place to
hear an objection for the first time — for you *and* for them. Five minutes
beforehand converts a public argument into a private adjustment.

---

## After

Within a day, send:

```text
DECIDED       what was decided, in one line
OWNER         who does what, by when
OPEN          what was not resolved, and who is finding out
NEXT          the review date
```

Four lines. This is the artifact that stops the meeting being relitigated in
three weeks, and it is the one nobody writes.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Opening with the agenda | The best minute of attention, spent on nothing |
| Method before the ask | They are still waiting for the point at minute nine |
| Reading the slides | The audience reads faster |
| A live demo with no backup video | You lose the room instead of ten seconds |
| Arguing with a hostile question | Name the disagreement instead |
| Inventing an answer | One invented answer costs the credibility of all the real ones |
| No pre-brief before a difficult decision | The objection arrives in public |
| No written follow-up | The decision is relitigated in three weeks |

---

## Exercises

1. Cut your last deck to five slides. What did you lose, and did it matter?
2. Rewrite every slide title as a finding with a number in it.
3. Record the 90-second backup video for your demo.
4. Write the hostile question you most fear, and script the answer using the
   name-the-assumption structure.
5. Send the four-line follow-up after your next meeting and see whether it gets
   referenced later.

---

**Next:** [Lesson 08 — The Writing Process](08-the-writing-process.md)
