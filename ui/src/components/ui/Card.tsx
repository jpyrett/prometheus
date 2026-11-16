import { useId, type ReactNode } from "react";

/** A card with the mockup's header: an 11 px capital title and a quiet right-hand side. */
export function Card({ title, right, badge, grow = false, className = "", style, titleId, children }: { title: string; right?: ReactNode; badge?: ReactNode; grow?: boolean; className?: string; style?: React.CSSProperties; titleId?: string; children: ReactNode }) {
  const generated = useId();
  const id = titleId ?? generated;
  return (
    <section className={`card${grow ? " grow" : ""}${className ? ` ${className}` : ""}`} style={style} aria-labelledby={id}>
      <header className="card-h">
        <h2 id={id}>{title}</h2>
        {badge}
        {right !== undefined && <div className="right">{right}</div>}
      </header>
      {children}
    </section>
  );
}
