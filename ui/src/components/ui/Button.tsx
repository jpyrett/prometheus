import type { ComponentProps } from "react";

export type ButtonVariant = "default" | "ghost" | "go" | "yes" | "no";

/** React 19 passes `ref` as an ordinary prop, so `...rest` forwards it. */
export function Button({ variant = "default", className = "", type = "button", ...rest }: ComponentProps<"button"> & { variant?: ButtonVariant }) {
  const v = variant === "default" ? "" : ` ${variant}`;
  return <button type={type} className={`btn${v}${className ? ` ${className}` : ""}`} {...rest} />;
}
