import type { InputHTMLAttributes, ReactNode } from "react";

export function Check({ label, fine = false, title, ...rest }: Omit<InputHTMLAttributes<HTMLInputElement>, "type"> & { label: ReactNode; fine?: boolean }) {
  return (
    <label className={`check${fine ? " fine" : ""}`} title={title}>
      <input type="checkbox" {...rest} />
      <span className="box" aria-hidden="true" />
      {label}
    </label>
  );
}
