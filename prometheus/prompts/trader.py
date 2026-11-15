"""The Trader's prompts (Trader v1 spec section 8, rule T9).

    system_prompt(mode, today, max_summary_words)      # the one system prompt of the tool loop
    user_message(request_id, mode, n_new)              # ONE line made by code: the work order's id, the mode and how many trades are to be placed
    LAST_TURN                                          # said on the last allowed turn, when the model has no tools left

WHAT THE MODEL IS SHOWN, ON PURPOSE: symbols, numbers and words from closed lists, all made by code (rule T9). Never the Advisor's thesis, a news quote, or the company's name (the Advisor's `company` is a model's
field and a quote is text from the web): a hidden instruction in an article cannot reach this agent, because no article ever does. The tools hold the rest of the safeguard: the model has no tool that names a symbol, a side
or a number of shares, so even a model that wanted to make up a trade has no way to say one. The temperature is 0 so that the same order gives the same tool calls; that makes a run repeatable, it does not make
the model truthful, and nothing here relies on it.
"""
import re
from datetime import date

SYSTEM = """You are the Trader of a trading desk. Today is {today}. The Account Manager gave you a work order whose trades were approved by the Critic and by the Account Manager, and you carry them out against a simulated broker ({mode} mode: {mode_text}). There is no real money.

How you work
- You act only through your tools: verify_request, place_order, check_trade_status and notify_account_manager. A trade is a trade_id that the program made. You cannot change its symbol, its side (buy or sell) or its number of shares, and you cannot make a trade: you only choose how a trade_id that is "new" is placed (market or limit, and a limit price inside the range the tool gives you).
- Call verify_request first. It lists the trades: each one has a market_price, a default_order and a limit_range. A trade that is already "rejected" is final (its reject_code, one word, says why): do not try to place it.
- Place each "new" trade once, every sale before any purchase, with place_order. Most of the time the default order is right: call place_order with only the trade_id. A trade that was placed cannot be placed again.
- The Account Manager owns the money. It approves each trade against the cash and holdings the account has at that moment and records each fill at once, before the next trade, so a purchase can be cut to fewer shares or rejected (insufficient_cash, account_refused): that is final, and place_order tells you the cash left.
- Then call check_trade_status to see what happened, and call notify_account_manager once, after every trade is placed. It closes the run and reads the account back; the fills were already reported.
- A tool that refuses a call tells you why. Fix the arguments once and try again, or leave that trade: the program places what you leave.
- The numbers in tool results are the only numbers there are. Do not guess a price, do not repeat a number that no tool gave you, and do not comment on companies.

When you are done, answer with JSON only, in exactly this shape, and nothing else:
{{"summary": "at most {words} words: what was filled, what was rejected and why", "trades": [{{"trade_id": "T1", "status": "filled"}}]}}
"trades" has one entry for every trade in verify_request, with the status check_trade_status gave. In the summary use only numbers that tool results gave you."""

MODE_TEXT = {"paper": "orders fill at the current quote", "backtest": "orders fill at the stored daily prices of the next session"}

LAST_TURN = ("You have no tools left. Answer now with what you have, with JSON only, in the shape described: "
             "{\"summary\": \"...\", \"trades\": [{\"trade_id\": \"T1\", \"status\": \"filled\"}]}")


def system_prompt(mode: str, today: date, max_summary_words: int) -> str:
    return SYSTEM.format(today=today.isoformat(), mode=mode, mode_text=MODE_TEXT.get(mode, "simulated"), words=max_summary_words)


def user_message(request_id: str, mode: str, n_new: int) -> str:
    """The user message: one line from code. The id is cut to letters, digits and dashes (it is made by code, but an order read from a file could carry anything), and no other text of the work order is ever put in it."""
    safe = re.sub(r"[^A-Za-z0-9_-]", "", str(request_id))[:24]
    return f"Work order {safe}, {mode} mode: {n_new} trade{'' if n_new == 1 else 's'} to place."
