import { Fragment, useRef, useState, type ReactNode } from "react";
import type { NewsItem, NewsView } from "../api/types";
import { Badge } from "./ui/Badge";
import { Card } from "./ui/Card";

/** A link only when it is a web address: a poisoned article must not be able to put `javascript:` behind a headline. */
export function safeUrl(url: string | null | undefined): string | null {
  return typeof url === "string" && /^https?:\/\//i.test(url.trim()) ? url.trim() : null;
}

const MARKS = /(\[\d+(?:\s*,\s*\d+)*\])/g;

const BOLD = /(\*\*[^*\n]+?\*\*)/g;
const BULLET = /^\s*[-*•]\s+/;

/** The summary as TEXT (entry 149, N9). The model writes a little Markdown: a line that starts with "- " or "* " is a bullet and `**words**` is bold; both are made into React elements, every
 * other character is shown as it is, and each `[n]` is linked to the article that carries that source number. Nothing in it is rendered as HTML: React escapes every string. */
export function SummaryText({ text, items, onJump }: { text: string; items: NewsItem[]; onJump: (number: number) => void }): ReactNode {
  const blocks: { bullet: boolean; lines: string[] }[] = [];
  for (const line of text.split("\n")) {
    if (!line.trim()) continue;
    const bullet = BULLET.test(line);
    const last = blocks[blocks.length - 1];
    if (last && last.bullet === bullet && bullet) last.lines.push(line.replace(BULLET, ""));
    else blocks.push({ bullet, lines: [bullet ? line.replace(BULLET, "") : line.trim()] });
  }
  return blocks.map((block, i) =>
    block.bullet ? (
      <ul className="news-bullets" key={i}>
        {block.lines.map((line, k) => (
          <li key={k}>
            <Inline text={line.trim()} items={items} onJump={onJump} />
          </li>
        ))}
      </ul>
    ) : (
      <p className="news-para" key={i}>
        <Inline text={block.lines[0]!} items={items} onJump={onJump} />
      </p>
    ),
  );
}

/** One line of the summary: `**bold**` as <strong>, the rest through the source marks. */
function Inline({ text, items, onJump }: { text: string; items: NewsItem[]; onJump: (number: number) => void }): ReactNode {
  return text.split(BOLD).map((part, i) =>
    /^\*\*[^*\n]+?\*\*$/.test(part) ? (
      <strong key={i}>
        <Marks text={part.slice(2, -2)} items={items} onJump={onJump} />
      </strong>
    ) : (
      <Fragment key={i}>
        <Marks text={part} items={items} onJump={onJump} />
      </Fragment>
    ),
  );
}

/** Each `[n]` linked to the article that carries that source number. */
function Marks({ text, items, onJump }: { text: string; items: NewsItem[]; onJump: (number: number) => void }): ReactNode {
  const byNumber = (n: number) => items.find((item) => item.numbers.includes(n));
  return text.split(MARKS).map((part, i) => {
    const inner = /^\[(\d+(?:\s*,\s*\d+)*)\]$/.exec(part);
    if (!inner) return <Fragment key={i}>{part}</Fragment>;
    const numbers = inner[1]!.split(",").map((n) => Number(n.trim()));
    return (
      <Fragment key={i}>
        [
        {numbers.map((n, k) => {
          const item = byNumber(n);
          return (
            <Fragment key={k}>
              {k > 0 && ", "}
              {item ? (
                <a className="news-mark" href={`#news-item-${item.number}`} onClick={(e) => (e.preventDefault(), onJump(item.number))} title={item.headline ?? undefined}>
                  {n}
                </a>
              ) : (
                n
              )}
            </Fragment>
          );
        })}
        ]
      </Fragment>
    );
  });
}

/** What the header says: "TSLA · Tesla Inc · last 7 days · stored news" (or "fetched now"). */
export function newsHeader(news: NewsView): string {
  const what = news.topic ? [news.topic] : news.stocks.flatMap((s) => (s.company ? [s.symbol, s.company] : [s.symbol]));
  const window = news.days ? `last ${news.days} day${news.days === 1 ? "" : "s"}` : null;
  return [...what, window, news.topic ? "stored news" : news.fetched ? "fetched now" : "stored news"].filter(Boolean).join(" · ");
}

/** The News view of the right column (entry 149, N9): the summary the Data Clerk wrote, the articles it came from, and what went wrong. It shows the news of the last request that asked only for the news. */
export function NewsPanel({ news }: { news: NewsView | null }) {
  const [lit, setLit] = useState<number | null>(null);
  const list = useRef<HTMLOListElement>(null);
  if (!news) {
    return (
      <Card title="News" grow className="news" right="from the Data Clerk">
        <div className="empty">
          <b>No news yet</b>
          Ask for a stock's news, e.g. "What's Tesla doing the last week?"
        </div>
      </Card>
    );
  }
  const jump = (number: number) => {
    setLit(number);
    const el = Array.from(list.current?.querySelectorAll<HTMLElement>("[data-number]") ?? []).find((node) => node.dataset.number === String(number));
    el?.scrollIntoView?.({ block: "nearest" });
    el?.focus();
  };
  return (
    <Card title="News" grow className="news" right={<Badge tone={news.fetched ? "pos" : "neutral"}>{news.topic ? "stored news" : news.fetched ? "fetched now" : "stored news"}</Badge>}>
      <div className="news-b scroll" role="region" tabIndex={0} aria-label="News summary and articles">
        <div className="news-head">{newsHeader(news)}</div>
        {news.fetch_note && (
          <div className="news-warn" role="status">
            {news.fetch_note}
          </div>
        )}
        {news.not_found.map((m) => (
          <div className="err" key={m.typed}>
            Not found: {m.typed}: {m.reason}
          </div>
        ))}
        {news.summary ? (
          <div className="news-summary">
            <SummaryText text={news.summary} items={news.items} onJump={jump} />
          </div>
        ) : (
          <div className="empty">
            <b>No summary</b>
            The Data Clerk gave no summary.
          </div>
        )}
        {news.no_supporting_text && <div className="hint">The stored news has nothing about this in the window.</div>}
        {news.items.length > 0 && (
          <>
            <h4 className="news-sec">Articles ({news.items.length}) · the ones the summary cites first</h4>
            <ol className="news-items" ref={list}>
              {news.items.map((item) => {
                const url = safeUrl(item.url);
                const headline = item.headline || "(no headline)";
                return (
                  <li key={item.number} id={`news-item-${item.number}`} data-number={item.number} tabIndex={-1} className={`news-item${item.cited ? " cited" : ""}${lit === item.number ? " lit" : ""}`}>
                    <span className="news-n num">[{item.numbers.join(", ")}]</span>
                    <div className="news-t">
                      {url ? (
                        <a href={url} target="_blank" rel="noopener noreferrer">
                          {headline}
                        </a>
                      ) : (
                        <span>{headline}</span>
                      )}
                      <div className="news-meta">
                        {[item.publisher, item.date, item.symbols.join(", ")].filter(Boolean).join(" · ")} {item.cited ? <Badge tone="info">cited</Badge> : <Badge>not cited</Badge>}
                      </div>
                    </div>
                  </li>
                );
              })}
            </ol>
          </>
        )}
        {news.problems.map((p) => (
          <div className="err" key={p}>
            {p}
          </div>
        ))}
        {news.notes.map((n) => (
          <div className="hint" key={n}>
            {n}
          </div>
        ))}
      </div>
    </Card>
  );
}
