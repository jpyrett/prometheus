// Dark or light: the first visit follows the system (index.html sets it before the first paint), the choice is remembered.
import { useCallback, useEffect, useState } from "react";

export type Theme = "dark" | "light";
const KEY = "prometheus.theme";

function current(): Theme {
  return document.documentElement.getAttribute("data-theme") === "light" ? "light" : "dark";
}

function chosen(): Theme | null {
  try {
    const v = localStorage.getItem(KEY);
    return v === "light" || v === "dark" ? v : null;
  } catch {
    return null;
  }
}

export function useTheme() {
  const [theme, setTheme] = useState<Theme>(current);

  useEffect(() => {
    // Until a choice is made the page follows the system as it changes.
    const query = window.matchMedia?.("(prefers-color-scheme: light)");
    if (!query) return;
    const follow = () => {
      if (chosen() !== null) return;
      const next: Theme = query.matches ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      setTheme(next);
    };
    query.addEventListener("change", follow);
    return () => query.removeEventListener("change", follow);
  }, []);

  const set = useCallback((next: Theme, remember: boolean) => {
    document.documentElement.setAttribute("data-theme", next);
    setTheme(next);
    if (remember) {
      try {
        localStorage.setItem(KEY, next);
      } catch {
        /* not remembered */
      }
    }
  }, []);

  const toggle = useCallback(() => set(current() === "dark" ? "light" : "dark", true), [set]);
  return { theme, toggle, set };
}
