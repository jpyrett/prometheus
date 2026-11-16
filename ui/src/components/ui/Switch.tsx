/** A two-state slider switch (the mockup's `.switch`): `role="switch"`, `aria-checked`, and Space and Enter flip it. The knob and track are the CSS's. */
export function Switch({ checked, onChange, label, disabled = false, id }: { checked: boolean; onChange: (checked: boolean) => void; label: string; disabled?: boolean; id?: string }) {
  return (
    <button
      type="button"
      id={id}
      role="switch"
      aria-checked={checked}
      aria-label={label}
      className={`switch${checked ? " on" : ""}`}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      // Enter and Space flip it once, in every browser: the key's own click is cancelled so it is not counted twice.
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          if (!disabled) onChange(!checked);
        }
      }}
      onKeyUp={(e) => {
        if (e.key === " ") e.preventDefault();
      }}
    >
      <i />
    </button>
  );
}
