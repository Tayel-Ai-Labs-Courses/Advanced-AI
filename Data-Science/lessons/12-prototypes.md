# Lesson 12 — Prototypes That Get a Decision

**Read before [lesson 08](08-shipping-the-model.md).**

**Goal:** put something in front of a stakeholder in an afternoon, and know
which rung of the ladder answers which question.

## What you will learn

- Four rungs, and what each one is for
- A stakeholder-facing page in eighteen lines
- Why a prototype is not a shipping decision
- The line between a demo and a product

---

## The ladder

```python
"""How long does each rung of the prototyping ladder take to build, and what
does it let you learn? Measured by lines of code, which is the only honest
proxy available."""
import textwrap

RUNGS = {
    "a printed table": '''
        preds = model.predict_proba(X)[:, 1]
        print(pd.DataFrame({"id": ids, "risk": preds}).nlargest(10, "risk"))
    ''',
    "a CSV someone opens": '''
        out = pd.DataFrame({"id": ids, "risk": preds, "action": actions})
        out.sort_values("risk", ascending=False).to_csv("call_list.csv", index=False)
    ''',
    "a Gradio page": '''
        import gradio as gr
        def score(tenure, logins, tickets, failures, fee, plan, country, age):
            row = pd.DataFrame([dict(tenure_days=tenure, logins_last_30d=logins,
                                     support_tickets_last_30d=tickets,
                                     payment_failures_last_90d=failures,
                                     monthly_fee=fee, plan=plan,
                                     country=country, age_band=age)])
            p = float(bundle["pipeline"].predict_proba(row[bundle["feature_order"]])[0, 1])
            return {"risk": round(p, 3),
                    "action": "call" if p >= bundle["threshold"] else "skip"}
        gr.Interface(fn=score,
                     inputs=[gr.Number(label="tenure days"), gr.Number(label="logins 30d"),
                             gr.Number(label="tickets"), gr.Number(label="payment failures"),
                             gr.Number(label="monthly fee"),
                             gr.Dropdown(["basic", "plus", "pro"], label="plan"),
                             gr.Dropdown(["EG", "SA", "AE", "MA"], label="country"),
                             gr.Dropdown(["18-29", "30-44", "45-59", "60+"], label="age")],
                     outputs="json", title="Churn risk").launch()
    ''',
    "a FastAPI service": '''
        from fastapi import FastAPI
        from pydantic import BaseModel, Field
        app = FastAPI()
        class Customer(BaseModel):
            tenure_days: int = Field(ge=0)
            logins_last_30d: int = Field(ge=0)
            support_tickets_last_30d: int = Field(ge=0)
            payment_failures_last_90d: int = Field(ge=0)
            monthly_fee: float = Field(gt=0)
            plan: str
            country: str
            age_band: str
        @app.post("/score")
        def score(c: Customer):
            row = pd.DataFrame([c.model_dump()])[bundle["feature_order"]]
            p = float(bundle["pipeline"].predict_proba(row)[0, 1])
            return {"probability": round(p, 4),
                    "action": "call" if p >= bundle["threshold"] else "skip",
                    "model": "churn-v1"}
    ''',
}
print(f"{'rung':<24}{'lines':>7}   {'who can use it':<25}{'what it answers'}")
answers = {
    "a printed table": ("you", "is the model sane?"),
    "a CSV someone opens": ("one colleague", "is the output useful?"),
    "a Gradio page": ("any stakeholder", "do they trust it?"),
    "a FastAPI service": ("another system", "can it be integrated?"),
}
for name, code in RUNGS.items():
    lines = len([l for l in textwrap.dedent(code).strip().split("\n") if l.strip()])
    who, ans = answers[name]
    print(f"{name:<24}{lines:>7}   {who:<25}{ans}")

```

```text
rung                      lines   who can use it           what it answers
a printed table               2   you                      is the model sane?
a CSV someone opens           2   one colleague            is the output useful?
a Gradio page                18   any stakeholder          do they trust it?
a FastAPI service            19   another system           can it be integrated?
```

**A page any stakeholder can use is eighteen lines.** That is the number worth
remembering, because most teams skip this rung entirely and go from a notebook
to a two-week integration — and discover at the end that the stakeholder wanted
something else.

Each rung answers a different question, and **you cannot skip to the answer you
want**:

- A printed table tells *you* the model is not broken.
- A CSV tells you whether the **output shape** is useful. Almost every project's
  first real feedback is "this is fine but I need the customer's phone number
  next to it", and a CSV surfaces that on day two rather than week six.
- A Gradio page tells you whether people **trust** it. Watch what they type:
  stakeholders always test the model on the three customers they know
  personally, and their reaction to those three decides the project.
- An API tells you whether it can be **integrated** — which is an engineering
  question, not a modelling one.

---

## Eighteen lines, running

```python
# no-run: opens a web server
import gradio as gr

def score(tenure, logins, plan):
    row = pd.DataFrame([dict(tenure_days=tenure, logins_last_30d=logins, plan=plan)])
    p = float(bundle["pipeline"].predict_proba(row[bundle["feature_order"]])[0, 1])
    return {"risk": round(p, 3),
            "action": "call" if p >= bundle["threshold"] else "skip"}

gr.Interface(
    fn=score,
    inputs=[gr.Number(label="tenure days"),
            gr.Number(label="logins in last 30 days"),
            gr.Dropdown(["basic", "plus", "pro"], label="plan")],
    outputs="json",
    title="Churn risk",
    description="Scores one customer. Threshold 0.192 comes from call capacity.",
).launch()
```

```text
gradio served HTTP 200 - page bytes: 17344
score(45, 1, 'basic') -> {'risk': 0.985, 'action': 'call'}
score(500, 25, 'pro') -> {'risk': 0.0, 'action': 'skip'}
```

Two details that are not decoration:

**The `description` states where the threshold came from.** Lesson 06 spent a
whole lesson deriving 0.192 from call capacity; a prototype that hides that
invites "can you make it more accurate?" instead of "can we make more calls?".

**The output is the action, not just the score.** Lesson 08's rule applies from
the first prototype: the reader should never have to remember the threshold.

### Gradio or Streamlit

| | Gradio | Streamlit |
|---|---|---|
| Best for | One function in, one result out | A page with several views |
| Lines for a single-model demo | ~18 | ~30 |
| Layout control | Limited | Good |
| Built-in sharing link | Yes (`share=True`) | No |
| Charts and tables | Basic | Strong |

For "show them the model", Gradio. For "a small internal tool with filters and
charts", Streamlit. Neither is a production front end, and both are excellent at
not being one.

---

## What a prototype is not

A demo proves the happy path. It proves nothing about the eight things that
decide whether the system works:

| The demo does not test | Where it is handled |
|---|---|
| Input validation | [Lesson 08](08-shipping-the-model.md) |
| Unseen categories | [Lesson 04](04-features-and-pipelines.md) |
| Latency at volume | [Lesson 08](08-shipping-the-model.md), LLM lesson 10 |
| The threshold under real capacity | [Lesson 06](06-evaluating-the-decision.md) |
| Drift | [Lesson 09](09-monitoring-and-drift.md) |
| Subgroups the model fails | [Lesson 10](10-limits-and-fairness.md) |
| Who may see the output | Data-Security lesson 01 |
| What happens when it is wrong | Your failure path |

**The most expensive sentence in this field is "the demo worked, so let's
ship".** A demo is one input chosen by the person who built it.

Two rules that keep prototypes honest:

1. **Never put real customer data in a shared prototype.** `share=True` creates
   a public URL. Use synthetic rows — Data-Security lesson 01's most common
   incident starts exactly here.
2. **Label it.** Put "PROTOTYPE — not monitored, not validated" in the title.
   Prototypes get bookmarked, and six months later someone is making decisions
   from one.

---

## The demo script

Fifteen minutes, in this order:

```text
1. The decision it supports        "who to call on Monday"        30 seconds
2. One realistic case              a customer they recognise      2 minutes
3. One edge case                   the model's actual limit       2 minutes
4. The number                      41,700 EGP/month, and the range 2 minutes
5. What it cannot do               the four limitations            2 minutes
6. What you need from them         the decision, with a date       2 minutes
7. Questions                                                       5 minutes
```

Step 3 is the one that builds trust. Showing a case where the model is
**wrong or unsure**, before they find it themselves, is the difference between
a stakeholder who believes your limitations section and one who now doubts
everything.

Communication lesson 07 covers the rest — including the backup video, because a
prototype that fails live costs you the room.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Going from notebook to integration | You find out in week six that the output shape was wrong |
| Skipping the CSV rung | The cheapest feedback you will ever get |
| Real customer data in a shared demo | A public URL with personal data on it |
| No "PROTOTYPE" label | It gets bookmarked and used for decisions |
| Hiding where the threshold came from | "Make it more accurate" instead of "make more calls" |
| Showing only cases that work | They find the broken one themselves, and stop trusting you |
| Treating demo success as readiness | Eight untested things stand between it and production |

---

## Exercises

1. Build the CSV rung for your current model and send it to one colleague. Write
   down the first thing they ask for.
2. Build the Gradio page in under 25 lines. Time yourself.
3. Find the edge case where your model is visibly unsure, and script how you
   would show it.
4. Add "PROTOTYPE — not monitored" to a demo you already have.
5. List which of the eight untested items your demo would pass today.

---

**Next:** [Lesson 08 — Shipping the Model](08-shipping-the-model.md), which
turns the fourth rung into something that can survive a Monday.
