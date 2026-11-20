"""Agentic security v1 (design choices entry 116; design/agentic_security_spec.md): policies, the Risk Score, the router and the Trust Score.

    policy.py   loads and checks `configs/security_v1.yaml`: every threshold, pattern list and message
    rules.py    the code rules (a user request, a document, a message to the user); a request rule is EXACT (blocks alone) or a HINT (the model decides; entry 121)
    scorer.py   the LLM Risk Scorer (role `guard`): it only labels, code takes the number
    router.py   `act(risk)` and `trust_score(...)`, the week 6 class example
    guard.py    the doors: `Guard.screen_request`, `screen_document`, `screen_message`, `check_rate`, `trust`
    audit.py    the append-only audit log (`logs/security_audit.jsonl`)
    rate.py     the per-account rate limit

The "risk" here is SECURITY risk only. Nothing in this package refuses, cuts or resizes a trade.
"""
