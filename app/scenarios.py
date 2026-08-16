"""Content for the synthetic corpus, grouped so answers match their questions.

The first version of the generator drew questions and answers from separate lists
and paired them at random, which produced traces where someone asked about
inference spend and got an answer about retry logic. That is fine for exercising
the export code and useless for reading, and the point of the inspector is that a
human can read it. So every scenario carries its own good answer, its own weak
answer, a truncated variant, and the tool the turn would plausibly call.

The weak answers are deliberately weak in the way real models are weak: fluent,
on topic, and empty. Refusals and hedges rather than nonsense, because that is
what a preference pair has to be able to separate.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Scenario:
    system: str
    question: str
    good: str
    weak: str
    truncated: str
    tool: tuple[str, dict] | None = None
    follow_ups: list[tuple[str, str, str]] = field(default_factory=list)


SUPPORT = ("You are a support agent for a logistics company. Be concise and never "
           "invent tracking numbers.")
CODING = ("You are a coding assistant embedded in a CI pipeline. Prefer minimal "
          "diffs and explain why a change is needed.")
ANALYTICS = ("You are an internal analytics copilot. Answer with numbers first, "
             "caveats second.")
SCHEDULING = ("You are a scheduling assistant with calendar and email access. "
              "Confirm before you write anything.")


SCENARIOS: list[Scenario] = [
    Scenario(
        system=SUPPORT,
        question="Where is order 4471 right now?",
        good=("Order 4471 left the Rotterdam hub at 06:12 today and is out for "
              "delivery. The carrier's own ETA is 17:40 local time. It was held "
              "overnight for a customs document, which cleared at 05:40."),
        weak=("I am not able to look up individual orders right now. You could try "
              "checking the tracking page, which usually has the most recent status."),
        truncated=("Order 4471 left the Rotterdam hub at 06:12 today. Before that it "
                   "sat in the customs queue overnight because the commercial invoice "
                   "was missing a HS code, which is the same issue that delayed two "
                   "other shipments on the same manifest, and looking at the pattern "
                   "across the last fortnight the"),
        tool=("search_orders", {"order_id": "4471"}),
        follow_ups=[
            ("Has this happened to this customer before?",
             "Twice in the last 90 days, both times on the same HS code. Their "
             "shipping partner is filing it as 8471 when it should be 8473.",
             "It is possible this has happened before, though I would need to look "
             "into the account history to say for certain."),
            ("Draft a message I can send them.",
             "Subject: 4471 delayed overnight, out for delivery today. Hi, your "
             "order cleared customs at 05:40 and is on the van, ETA 17:40. The hold "
             "was a missing HS code on the invoice, the third time on this code, so "
             "we are raising it with your shipping partner.",
             "Here is a draft: Dear customer, we apologise for the delay with your "
             "order and appreciate your patience. Please let us know if you have "
             "any further questions."),
        ],
    ),
    Scenario(
        system=CODING,
        question="Why did the nightly build fail on the payments service?",
        good=("The failure is in the migration step, not the tests. "
              "`payments/0043_add_idx` creates an index on `settled_at`, but that "
              "column is only added in `0044`. On a database that already has the "
              "column it passes, on a fresh one it does not, which is why it only "
              "fails on nightly. Reorder the two migrations."),
        weak=("It looks like something went wrong during the build. I would suggest "
              "reviewing the logs to see if any particular step stands out as the "
              "cause of the failure."),
        truncated=("The failure is in the migration step rather than the test suite. "
                   "Walking through it, the nightly job builds from a clean database, "
                   "so every migration runs in order against an empty schema, and the "
                   "index creation in 0043 refers to a column that does not exist yet "
                   "because the column is added in 0044, which means the ordering is "
                   "wrong and the"),
        tool=("get_build_log", {"service": "payments", "run": 8821}),
        follow_ups=[
            ("Can you show the fix as a diff?",
             "Rename the files so 0043 adds the column and 0044 adds the index. "
             "Nothing inside them changes, only the ordering and the dependency "
             "line in 0044.",
             "You would need to modify the migration files so that they run in the "
             "correct order relative to one another."),
            ("Will that break anyone already on the old ordering?",
             "No. Django records applied migrations by name, so anyone who already "
             "ran 0043 keeps it recorded and the reordered file is a no-op for them. "
             "Only fresh databases take the new path.",
             "There may be some risk to existing environments, so it would be wise "
             "to test the change carefully before deploying."),
        ],
    ),
    Scenario(
        system=ANALYTICS,
        question="Summarise last week's churn by plan tier.",
        good=("Pro 2.1%, Team 1.4%, Enterprise 0.3%. Pro is up from 1.6% the week "
              "before. Almost all of the increase is accounts under three months "
              "old, so this reads as onboarding rather than a pricing problem. "
              "Caveat: two Enterprise accounts are mid-renewal and not counted yet."),
        weak=("Churn varies across plan tiers and can be influenced by a number of "
              "factors including pricing, onboarding experience and product fit. "
              "Reviewing the data by segment would help clarify the picture."),
        truncated=("Pro 2.1%, Team 1.4%, Enterprise 0.3%. Before drawing anything "
                   "from that it is worth splitting the number three ways, first by "
                   "plan tier, then by account age, then by whether the account "
                   "arrived through self-serve or through sales, because the headline "
                   "figure moves a lot once accounts younger than"),
        tool=("run_sql", {"query": "select plan, count(*) from subs where churned_at "
                                   ">= date('now','-7 day') group by 1"}),
        follow_ups=[
            ("Show me the query you used.",
             "select plan, count(*) filter (where churned_at >= current_date - 7) "
             "/ count(*)::float from subscriptions group by plan. Churn is measured "
             "against accounts active at the start of the window, not the end.",
             "I used a standard aggregation query grouping subscriptions by plan "
             "tier over the relevant time period."),
            ("Do the same for the previous quarter.",
             "Q2: Pro 1.7%, Team 1.2%, Enterprise 0.4%. Pro has drifted up about "
             "0.4 points across the quarter, steadily rather than in a jump, which "
             "again points at onboarding.",
             "The previous quarter shows broadly similar trends across the tiers "
             "with some variation month to month."),
        ],
    ),
    Scenario(
        system=CODING,
        question="The retry logic in queue.py looks wrong to me, does it double-process?",
        good=("Yes, it can. The SQS visibility timeout is 30s but `handle_batch` can "
              "run to about 45s on a large batch, so the message becomes visible "
              "again and a second consumer picks it up while the first is still "
              "working. Either raise the timeout above the worst-case runtime or "
              "make the handler idempotent on message id. Idempotency is the safer "
              "of the two, since the runtime will grow again."),
        weak=("The retry logic may or may not double-process depending on your "
              "configuration and the volume of messages being handled at the time."),
        truncated=("Yes, it can double-process, and the reason is a mismatch between "
                   "two timeouts that are set in different places. The visibility "
                   "timeout on the queue is thirty seconds, which is the window in "
                   "which a consumer has to delete the message before it becomes "
                   "visible again, but the handler itself has no such bound and on a "
                   "large batch it will happily run for"),
        tool=("read_file", {"path": "services/queue.py"}),
        follow_ups=[
            ("Which fix would you take?",
             "Idempotency. Raising the timeout only moves the cliff, and the batch "
             "size is already growing month on month. A processed-message table with "
             "a TTL is about fifteen lines and it stops caring how long the handler "
             "takes.",
             "Both options have merits. It depends on your priorities and the "
             "constraints of your system."),
        ],
    ),
    Scenario(
        system=ANALYTICS,
        question="How much did we spend on inference last month, split by model?",
        good=("$4,180 total. gpt-4o-mini was 61% of calls but 9% of spend. gpt-4o was "
              "12% of calls and 58% of spend. claude-sonnet-4 was 8% of calls and 27% "
              "of spend. The routing is doing its job on volume, but the expensive "
              "tail is where the money is."),
        weak=("Inference costs can add up across different models. Reviewing your "
              "usage dashboard would give you a breakdown of spend by model for the "
              "period you are interested in."),
        truncated=("$4,180 across all models last month. The split is more useful "
                   "than the total, because the share of calls and the share of spend "
                   "point in opposite directions: the cheap model carries most of the "
                   "traffic while the expensive ones carry most of the cost, and once "
                   "you break it down by route rather than by model the"),
        tool=("run_sql", {"query": "select model, sum(cost_usd) from traces "
                                   "group by 1 order by 2 desc"}),
        follow_ups=[
            ("Where would you cut first?",
             "The retries. 19% of gpt-4o calls last month were retries of a cheaper "
             "model that had already failed, so we paid twice for those turns. "
             "Routing the known-hard prompt shapes to gpt-4o first would cost less "
             "than routing them there second.",
             "There are several areas where costs could potentially be reduced, "
             "depending on your usage patterns."),
        ],
    ),
    Scenario(
        system=SCHEDULING,
        question="Move my Thursday one-to-ones to Friday morning.",
        good=("Four one-to-ones on Thursday. Three move cleanly to Friday 09:00, "
              "09:30 and 10:00. The fourth, with Priya, clashes with the design "
              "review she owns at 10:30, so I have left it. Shall I send the updates "
              "for the three and ask Priya separately?"),
        weak=("I can help with rescheduling meetings. Please confirm which meetings "
              "you would like to move and I will do my best to assist."),
        truncated=("There are four one-to-ones on Thursday and Friday morning has "
                   "room for three of them without touching anything else, but the "
                   "fourth is harder because Priya owns the design review at half "
                   "past ten and moving her one-to-one into that slot would either "
                   "clash directly or push the review, which affects six other people "
                   "and the"),
        tool=("list_calendar", {"range": "2026-08-20/2026-08-21"}),
        follow_ups=[
            ("Yes, send them.",
             "Sent three updates. Priya's is untouched and I have drafted a note "
             "asking her for a slot after 11:00 on Friday, not sent yet.",
             "I have gone ahead and made the requested changes to your calendar."),
        ],
    ),
    Scenario(
        system=SUPPORT,
        question="Which customers opened a ticket twice in the last 30 days?",
        good=("Nine accounts. Six of the nine opened both tickets about the same "
              "thing, and five of those six are about the customs hold on the "
              "Rotterdam route. That is one problem showing up as nine tickets, not "
              "nine unhappy customers."),
        weak=("Several customers have opened multiple tickets recently. Looking at "
              "your ticketing system would let you identify which accounts have "
              "been in contact more than once."),
        truncated=("Nine accounts opened two or more tickets in the window. The raw "
                   "count is less interesting than the overlap, because when you "
                   "group the tickets by subject rather than by account most of them "
                   "collapse into a single underlying issue, and the ones that do not "
                   "collapse are mostly billing questions from accounts that"),
        tool=("run_sql", {"query": "select account_id, count(*) from tickets "
                                   "where opened_at > now() - interval '30 day' "
                                   "group by 1 having count(*) > 1"}),
        follow_ups=[
            ("Which is the most urgent?",
             "Bergman Logistics. Both tickets are the customs hold, the second one "
             "came in after our reply, and they have a renewal on 1 September.",
             "Urgency depends on your criteria, but accounts with repeated contact "
             "are generally worth prioritising."),
        ],
    ),
    Scenario(
        system=ANALYTICS,
        question="Find the duplicate supplier records and tell me which to keep.",
        good=("Fourteen duplicate pairs. Twelve are the same supplier with and "
              "without a legal suffix, keep the one with the tax id populated. Two "
              "are genuinely different entities that share a trading name, do not "
              "merge those: Nordwind GmbH in Hamburg and Nordwind BV in Rotterdam."),
        weak=("Duplicate records can usually be identified by comparing names and "
              "addresses. Once identified, it is generally best to keep the most "
              "complete record."),
        truncated=("There are fourteen pairs that look like duplicates on a name "
                   "match, but name matching alone will merge two suppliers that are "
                   "genuinely separate legal entities trading under the same name in "
                   "different countries, so the useful comparison is name plus tax id "
                   "plus billing country, and once you apply that the"),
        tool=("run_sql", {"query": "select name, count(*) from suppliers group by 1 "
                                   "having count(*) > 1"}),
    ),
]

TOOL_ERRORS = [
    "upstream timeout after 30s",
    "permission denied for table subs",
    "404 no build found for run 8821",
    "rate limited, retry after 12s",
    "connection reset by peer",
]
