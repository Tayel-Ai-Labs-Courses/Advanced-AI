# Lesson 08 — Prompts in Production

**Goal:** treat a prompt as a production artefact, with a version, an owner and
a test.

## What you will learn

- Where a prompt lives, and why not in the code
- The template rules that prevent most incidents
- What to log on every call
- When the answer is to stop prompting

---

## A prompt is code

It has a version, an input contract, an output contract, a failure mode and a
cost per call. The only unusual thing about it is that it is a string — and
that is exactly what tempts teams to treat it as a configuration detail.

```text
WRONG                                RIGHT
a string in the service              a file, versioned in git
edited to fix a complaint            changed through review, with a measurement
no tests                             an eval set that runs in CI
no version in the logs               the version in every logged response
anyone can change it                 an owner, like any other component
```

Every row maps to something [MLOps](../../MLOps/) measured for models. A prompt
is a model artefact with a different file extension, and the
[CI gate](../../MLOps/lessons/04-ci-gate.md) arithmetic — no gate 90,120
EGP/month against 16,020 with one — applies to it unchanged.

---

## Where it lives

```text
prompts/
  classify_review/
    v7.txt               the prompt, with {placeholders}
    contract.json        allowed outputs, the escape hatch
    eval.jsonl           300 labelled examples (lesson 06)
    CHANGELOG.md         every version, with n and p
```

Four properties that make this worth the directory:

**The prompt is a file, not a literal.** You can diff it, review it, and blame
it. A prompt embedded in a service is invisible in a pull request.

**The eval set sits next to it.** The prompt is disposable; the eval set is the
asset ([lesson 06](06-the-iteration-loop.md)). Keeping them together is what
makes a change testable by whoever makes it next.

**The contract is machine-readable**, so the parser and the gate read the same
file rather than two divergent copies of the same assumption.

**The changelog explains the weird parts.** Every mature prompt has a line that
looks unnecessary and is load-bearing. Without an entry saying why, someone
deletes it in a cleanup, and the bug comes back in a quarter.

---

## Template rules

```python
# no-run
TEMPLATE = """Classify the review sentiment.

Review: the staff were rude
Sentiment: negative

Review: {review}
Sentiment:"""

def build(review: str) -> str:
    review = review.strip()
    if len(review) > MAX_CHARS:                  # 1. bound the input
        review = review[:MAX_CHARS]
    review = review.replace("\n", " ")           # 2. flatten the structure
    return TEMPLATE.format(review=review)        # 3. one substitution point
```

Three rules, each preventing a real incident:

**Bound every injected value.** An unbounded field means one customer with a
20,000-character review can blow the context window, truncate your instructions
from the front, and cost fifty times the usual tokens on that call.

**Flatten structure out of untrusted values.** Replacing newlines does not stop
[lesson 07](07-untrusted-text.md)'s attacks — nothing at this layer does — but
it removes the cheapest version of the fake label line. Input validation, not a
boundary.

**One substitution point per value, in one function.** The moment two places
build the same prompt, they drift, and you are debugging a difference you
cannot see. This is the training/serving skew of
[MLOps 07](../../MLOps/lessons/07-monitoring.md) in a different costume: there,
a column swap took AUC from 0.9001 to 0.2135.

And one rule about the template itself: **static first, dynamic last**
([lesson 05](05-what-a-prompt-costs.md)), so the prefix caches and the answer
comes next.

---

## What to log

```text
EVERY CALL     prompt version          which string produced this
               model and parameters    name, temperature, max tokens
               prompt tokens           lesson 05's cost metric
               output tokens
               latency
               parse ok?               lesson 02's first metric
               the output              or a hash of it, if sensitive

NEVER          the raw prompt including customer text, in plaintext logs,
               with no retention limit. It is personal data now.
               See Data-Security-for-AI 02.
```

`parse ok?` is the one that earns its place. It is a number you can put a
threshold under: **a parse rate drifting from 99% to 94% is the earliest visible
sign that something upstream changed** — a new input source, a model update, a
template edit. [MLOps 07](../../MLOps/lessons/07-monitoring.md) has the
threshold arithmetic; use `sqrt(p(1-p)/n)` and alarm at three standard errors,
not at a number someone liked.

The prompt version in every log line is what makes an incident diagnosable. Six
weeks later, "did this get worse after v7?" is a query rather than an argument.

---

## The release checklist

- [ ] The eval set has at least ~300 examples ([lesson 06](06-the-iteration-loop.md))
- [ ] Parse rate measured separately from accuracy ([lesson 02](02-the-output-contract.md))
- [ ] The contract has an escape hatch (`unclear`) and the code handles it
- [ ] The change was **one** change, with an n and a p in the changelog
- [ ] Prompt tokens per call measured, and the examples justified ([lesson 05](05-what-a-prompt-costs.md))
- [ ] Untrusted values bounded, flattened, and built in one function
- [ ] Nothing irreversible happens on model output without a code check ([lesson 07](07-untrusted-text.md))
- [ ] The version is in the logs and in the response
- [ ] Rollback is changing one file back, and someone has done it
- [ ] A named owner

The gate for all of it is the same shape as a model's: **run the eval set in CI
and fail the change if it regresses.** A prompt without that is a model shipping
with no test, which is what [MLOps 04](../../MLOps/lessons/04-ci-gate.md)
priced at 5.6x the cost.

---

## When to stop prompting

The most valuable judgement in this course, and the one that takes longest to
learn. Stop when any of these is true:

| Signal | What it means | Go to |
|---|---|---|
| Output is well-formed and wrong | Capability gap, not a prompt gap | [Lesson 04](04-breaking-the-task-up.md) |
| The gain you need is under 5 points | 1,247 examples to even see it | [Lesson 06](06-the-iteration-loop.md) |
| The model lacks facts, not ability | Retrieval | [LLM 06](../../LLM-and-GenAI/lessons/06-rag.md) |
| You have thousands of labels | Fine-tune, or a classifier | [LLM 08](../../LLM-and-GenAI/lessons/08-prompt-rag-or-finetune.md) |
| The task is narrow and labelled | Logistic regression, four minutes | [Machine-Learning](../../Machine-Learning/) |
| The prompt is on its twelfth version | You are fitting the eval set, not improving | [Lesson 06](06-the-iteration-loop.md) |
| The required accuracy is unreachable | Do not build it | [Data-Science 02](../../Data-Science/lessons/02-framing-the-problem.md) |

The last row is a real answer and it is the one nobody gives.
[Data-Science 01](../../Data-Science/lessons/01-what-data-science-is.md)'s
finding — that a model can tie a do-nothing baseline on accuracy and still be
worth 41,699 EGP a month — has a mirror image: **a prompt can be impressive in a
demo and worth nothing, and the only way to know is the arithmetic in
[Data-Science 06](../../Data-Science/lessons/06-evaluating-the-decision.md).**

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| The prompt as a string literal in the service | Invisible in review, undiffable, untestable |
| No eval set in CI | A prompt is a model with no test |
| Unbounded injected values | One long input truncates your instructions |
| Two code paths building the same prompt | They drift; you debug an invisible difference |
| No prompt version in the logs | Incidents become arguments |
| Logging raw customer text forever | That is personal data with no retention policy |
| No escape hatch in the contract | You forced a guess ([lesson 02](02-the-output-contract.md)) |
| Version twelve of the same prompt | You are fitting the eval set |
| Never asking whether to stop | The best answer is sometimes a classifier, or nothing |

---

## Exercises

1. Move one prompt out of your code into a file with a version.
2. Write its contract as JSON and make the parser read it.
3. Add its eval set to CI. Make a PR that worsens the prompt; watch it fail.
4. Add prompt version, token counts and parse-ok to your logs.
5. Compute the alarm threshold for your parse rate.
6. For one prompt, work through the stop-prompting table. Which row are you in?

---

## Where to go next

| Next | Why |
|---|---|
| [Project 23](../Project-23/) | Build one prompt properly, end to end |
| [LLM-and-GenAI](../../LLM-and-GenAI/) | Tokens, decoding, RAG, evaluation, shipping |
| [AI-Agents](../../AI-Agents/) | What happens when a prompt is given powers |
| [Advanced-Prompt-Engineering](https://github.com/Tayel-Ai-Labs-Courses/Advanced-Prompt-Engineering) | The technique catalogue, once you can measure |
| [MLOps](../../MLOps/) | The gate, the logs and the rollback, for real |
