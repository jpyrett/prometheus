import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { replayEvents } from "../state/runReducer";
import { approveQuestion, bobEvents, clarifyQuestion, confirmQuestion, fundingQuestion, missingQuestion, raiseFundsQuestion } from "../test/fixtures";
import { QuestionCard } from "./QuestionCard";

function setup(question: typeof confirmQuestion, row?: Parameters<typeof QuestionCard>[0]["row"]) {
  const answer = vi.fn().mockResolvedValue(undefined);
  render(<QuestionCard question={question} answer={answer} row={row} />);
  return answer;
}

describe("QuestionCard: confirm (recorded from the replay of a direct order in Ask mode)", () => {
  it("shows what was understood and answers yes or no, with focus on the first button", () => {
    const answer = setup(confirmQuestion);
    expect(screen.getByText(/^I understood: buy 50 shares of MSFT/).textContent?.endsWith("Go ahead?")).toBe(true);
    expect(document.activeElement).toBe(screen.getByRole("button", { name: "Yes, go ahead" }));
    fireEvent.click(screen.getByRole("button", { name: "Yes, go ahead" }));
    expect(answer).toHaveBeenCalledWith(confirmQuestion.question_id, true);
  });

  it("No, stop answers false, and both buttons lock once an answer is on its way", () => {
    const answer = setup(confirmQuestion);
    fireEvent.click(screen.getByRole("button", { name: "No, stop" }));
    expect(answer).toHaveBeenCalledWith(confirmQuestion.question_id, false);
    expect((screen.getByRole("button", { name: "Yes, go ahead" }) as HTMLButtonElement).disabled).toBe(true);
  });
});

describe("QuestionCard: clarify (how much, from the Account Manager's real question and facts)", () => {
  it("shows the question, its number, the facts in a monospace block, and Send answers the typed text; Enter sends", () => {
    const answer = setup(clarifyQuestion);
    expect(screen.getByText("ACCOUNT MANAGER ASKS · 1 OF 3")).toBeTruthy();
    expect(screen.getByText(/How much would you like to buy\?/)).toBeTruthy();
    const facts = document.querySelector(".qcard pre");
    expect(facts?.textContent).toContain("Approved to buy:");
    expect(facts?.textContent?.split("\n").length).toBe(clarifyQuestion.facts?.length);
    const box = screen.getByLabelText("Your answer") as HTMLInputElement;
    expect(document.activeElement).toBe(box);
    expect((screen.getByRole("button", { name: "Send" }) as HTMLButtonElement).disabled).toBe(true);
    fireEvent.change(box, { target: { value: "20% of my account in total" } });
    fireEvent.submit(box.closest("form")!); // Enter in the box submits the form
    expect(answer).toHaveBeenCalledWith(clarifyQuestion.question_id, "20% of my account in total");
  });
});

describe("QuestionCard: approve (the Critic's own scorecard of Bob Jones' XOM, 1 of 2)", () => {
  const row = replayEvents(bobEvents).workOrder!.rows[0];

  it("says what would be bought, lists the whole proposal on the first card, and shows the Critic's verdict", () => {
    setup(approveQuestion, row);
    expect(screen.getByText("APPROVE TRADE 1 OF 2")).toBeTruthy();
    const proposal = screen.getByLabelText("The whole proposal");
    expect(proposal.textContent).toContain("XOM");
    expect(proposal.textContent).toContain("EOG");
    const sentence = document.querySelector(".qcard p")!.textContent!;
    expect(sentence).toContain("Buy 8 shares of XOM, about $1,364.72 of $15,000.00 cash.");
    expect(sentence).toContain("Critic: APPROVE 86 (B).");
  });

  it("Approve and Deny answer true and false", () => {
    const answer = setup(approveQuestion, row);
    fireEvent.click(screen.getByRole("button", { name: "Deny" }));
    expect(answer).toHaveBeenCalledWith(approveQuestion.question_id, false);
  });

  it("See the scorecard unfolds the card's own text, and folds it again", () => {
    setup(approveQuestion, row);
    expect(screen.queryByLabelText("The Critic's scorecard")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "See the scorecard" }));
    const card = screen.getByLabelText("The Critic's scorecard");
    expect(card.textContent).toContain("APPROVE score 86/100 (B)");
    expect(card.textContent).toBe(approveQuestion.scorecard);
    fireEvent.click(screen.getByRole("button", { name: "Hide the scorecard" }));
    expect(screen.queryByLabelText("The Critic's scorecard")).toBeNull();
  });

  it("leaves the proposal out from the second card on", () => {
    setup({ ...approveQuestion, n: 2, title: "Approve trade 2 of 2" }, row);
    expect(screen.queryByLabelText("The whole proposal")).toBeNull();
  });
});

describe("QuestionCard: a stock the Data Clerk did not find (recorded from the live Ask-mode session of Jacob's request, Chick-fil-a)", () => {
  it("says what is missing and why, shows what stays on the order, and its two buttons answer yes and no", () => {
    const answer = setup(missingQuestion);
    expect(screen.getByText("ACCOUNT MANAGER ASKS")).toBeTruthy();                                       // one question: no "1 OF 1"
    const sentence = document.querySelector(".qcard p")!.textContent!;
    expect(sentence).toBe(missingQuestion.question);                                                     // the Account Manager's own sentence, whole
    expect(sentence).toContain("Chick-fil-a was not found");
    expect(document.querySelector(".qcard p b")!.textContent).toBe("Chick-fil-a");                       // the stock as the user typed it stands out
    const block = screen.getByLabelText("What stays on the order");
    expect(block.textContent).toContain("F: Ford Motor Co (F)");
    expect(block.textContent).toContain("Google: Alphabet Inc. (GOOGL), the listed owner of what you typed");
    expect(document.activeElement).toBe(screen.getByRole("button", { name: "OK, continue without it" }));
    fireEvent.click(screen.getByRole("button", { name: "OK, continue without it" }));
    expect(answer).toHaveBeenCalledWith(missingQuestion.question_id, true);
    expect((screen.getByRole("button", { name: "Stop the order" }) as HTMLButtonElement).disabled).toBe(true);   // both lock once an answer is on its way
  });

  it("Stop the order answers false, and several questions say which one this is", () => {
    const answer = setup({ ...missingQuestion, n: 2, of: 3 });
    expect(screen.getByText("ACCOUNT MANAGER ASKS · 2 OF 3")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Stop the order" }));
    expect(answer).toHaveBeenCalledWith(missingQuestion.question_id, false);
  });
});


describe("QuestionCard: raising funds for a purchase the cash does not cover (entry 142; the numbers of Jacob's real request on a throwaway copy of his account)", () => {
  it("the question says how short the cash is in the Account Manager's own sentence, lists what would be bought, and its buttons answer yes and no", () => {
    const answer = setup(raiseFundsQuestion);
    expect(screen.getByText("ACCOUNT MANAGER ASKS · RAISE FUNDS?")).toBeTruthy();
    expect(document.querySelector(".qcard p")!.textContent).toBe("You have $60.14 available; 2 ANT at $88.91 cost $177.82, $117.68 short. Shall I raise the money by selling some of your holdings? I will show you what I would sell first.");
    expect(screen.getByLabelText("What would be bought").textContent).toContain("ANT");
    expect(screen.getByLabelText("What would be bought").textContent).toContain("$177.82");
    expect(document.activeElement).toBe(screen.getByRole("button", { name: "Yes, raise the money" }));
    fireEvent.click(screen.getByRole("button", { name: "Yes, raise the money" }));
    expect(answer).toHaveBeenCalledWith(raiseFundsQuestion.question_id, true);
    expect((screen.getByRole("button", { name: "No" }) as HTMLButtonElement).disabled).toBe(true); // both lock once an answer is on its way
    expect(screen.getByText("Nothing is sold without your yes. No answer means no.")).toBeTruthy();
  });

  it("No answers false", () => {
    const answer = setup(raiseFundsQuestion);
    fireEvent.click(screen.getByRole("button", { name: "No" }));
    expect(answer).toHaveBeenCalledWith(raiseFundsQuestion.question_id, false);
  });

  it("the plan shows which holding is sold, how many shares, at what price, the gain, and the cash before and after; Sell, then buy answers yes and No, sell nothing answers no", () => {
    const answer = setup(fundingQuestion);
    expect(screen.getByText("APPROVE THE SALES")).toBeTruthy();
    expect(document.querySelector(".qcard p")!.textContent).toBe("To buy 2 shares of ANT, with cash of $60.14, these holdings would be sold first:");
    const sells = screen.getByLabelText("The holdings that would be sold").textContent!;
    expect(sells).toContain("DPZ");
    expect(sells).toContain("Sell 1 at $348.61");
    expect(sells).toContain("$348.61, gain $6.89");
    expect(document.querySelector(".qcard .cap")!.textContent).toBe("The cash after the sales and the purchase would be about $230.93.");
    fireEvent.click(screen.getByRole("button", { name: "Sell, then buy" }));
    expect(answer).toHaveBeenCalledWith(fundingQuestion.question_id, true);
    expect((screen.getByRole("button", { name: "No, sell nothing" }) as HTMLButtonElement).disabled).toBe(true);
  });

  it("No, sell nothing answers false", () => {
    const answer = setup(fundingQuestion);
    fireEvent.click(screen.getByRole("button", { name: "No, sell nothing" }));
    expect(answer).toHaveBeenCalledWith(fundingQuestion.question_id, false);
  });
});
