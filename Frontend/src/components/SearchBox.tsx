import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Search, X, CornerDownLeft } from "lucide-react";
import { api, type SearchResult } from "@/lib/api";
import { cn } from "@/lib/utils";

interface Props {
  onSelect: (r: SearchResult) => void;
}

export function SearchBox({ onSelect }: Props) {
  const [value, setValue] = useState("");
  const [debounced, setDebounced] = useState("");
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Debounce
  useEffect(() => {
    const t = setTimeout(() => setDebounced(value.trim()), 300);
    return () => clearTimeout(t);
  }, [value]);

  // Keyboard: "/" focus, Esc clear
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "/" && document.activeElement?.tagName !== "INPUT") {
        e.preventDefault();
        inputRef.current?.focus();
      } else if (e.key === "Escape") {
        if (document.activeElement === inputRef.current) {
          setValue("");
          setOpen(false);
        }
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // Close on outside click
  useEffect(() => {
    function onDoc(e: MouseEvent) {
      if (!containerRef.current?.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, []);

  const enabled = debounced.length >= 2;
  const { data, isFetching } = useQuery({
    queryKey: ["search", debounced],
    queryFn: ({ signal }) => api.search(debounced, 10, signal),
    enabled,
    staleTime: 60_000,
  });

  const results = data?.results ?? [];

  function handleSelect(r: SearchResult) {
    onSelect(r);
    setValue(r.title);
    setOpen(false);
    inputRef.current?.blur();
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (!open || results.length === 0) return;
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActive((a) => Math.min(a + 1, results.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActive((a) => Math.max(a - 1, 0));
    } else if (e.key === "Enter") {
      e.preventDefault();
      handleSelect(results[active]);
    }
  }

  return (
    <div ref={containerRef} className="relative w-full">
      <div
        className={cn(
          "flex items-center gap-3 rounded-2xl border bg-card/70 px-4 py-3.5 backdrop-blur-sm transition",
          open && "border-[hsl(var(--primary)/0.4)]",
        )}
        style={{ borderColor: open ? undefined : "hsl(0 0% 100% / 0.1)" }}
      >
        <Search className="h-5 w-5 text-muted-foreground" />
        <input
          ref={inputRef}
          value={value}
          onChange={(e) => {
            setValue(e.target.value);
            setOpen(true);
            setActive(0);
          }}
          onFocus={() => setOpen(true)}
          onKeyDown={onKeyDown}
          placeholder="Search products by title — try 'binoculars' or 'drill'"
          className="flex-1 bg-transparent text-base outline-none placeholder:text-muted-foreground"
          aria-label="Search products"
        />
        {value && (
          <button
            type="button"
            onClick={() => {
              setValue("");
              inputRef.current?.focus();
            }}
            className="rounded-md p-1 text-muted-foreground hover:text-foreground"
            aria-label="Clear search"
          >
            <X className="h-4 w-4" />
          </button>
        )}
        <kbd className="hidden rounded border bg-muted px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground sm:inline-block"
             style={{ borderColor: "hsl(0 0% 100% / 0.1)" }}>
          /
        </kbd>
      </div>

      {open && enabled && (
        <div
          className="absolute left-0 right-0 top-full z-30 mt-2 max-h-80 overflow-auto rounded-2xl border bg-popover/95 p-1.5 shadow-xl backdrop-blur-md"
          style={{ borderColor: "hsl(0 0% 100% / 0.1)" }}
        >
          {isFetching && results.length === 0 && (
            <div className="px-3 py-6 text-center text-sm text-muted-foreground">
              Searching catalogue…
            </div>
          )}
          {!isFetching && results.length === 0 && (
            <div className="px-3 py-6 text-center text-sm text-muted-foreground">
              No matches for <span className="font-mono">"{debounced}"</span>
            </div>
          )}
          {results.map((r, i) => (
            <button
              key={r.asin}
              onMouseEnter={() => setActive(i)}
              onClick={() => handleSelect(r)}
              className={cn(
                "flex w-full items-center justify-between gap-3 rounded-xl px-3 py-2.5 text-left",
                active === i ? "bg-accent/40" : "hover:bg-accent/30",
              )}
            >
              <span className="line-clamp-1 text-sm">{r.title}</span>
              <span className="flex shrink-0 items-center gap-2">
                <span className="font-mono text-[11px] text-muted-foreground">{r.asin}</span>
                {active === i && <CornerDownLeft className="h-3.5 w-3.5 text-[hsl(var(--primary))]" />}
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
