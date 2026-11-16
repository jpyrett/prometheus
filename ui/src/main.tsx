import "@fontsource-variable/inter";
import "@fontsource-variable/jetbrains-mono";
import { lazy, StrictMode, Suspense } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./styles/tokens.css";
import "./styles/base.css";
import "./styles/components.css";

// ?states draws the sheet of component states (a development view, see StatesPage.tsx); everything else is the page.
const StatesPage = lazy(() => import("./StatesPage"));
const states = new URLSearchParams(window.location.search).has("states");

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    {states ? (
      <Suspense fallback={null}>
        <StatesPage />
      </Suspense>
    ) : (
      <App />
    )}
  </StrictMode>,
);
