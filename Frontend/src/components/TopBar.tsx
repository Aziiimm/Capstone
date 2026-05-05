import { Cpu } from "lucide-react";
import { StatusPill } from "./StatusPill";
import { ThemeToggle } from "./ThemeToggle";

export function TopBar() {
  return (
    <header
      className="sticky top-0 z-40 border-b backdrop-blur-md"
      style={{
        borderColor: "hsl(0 0% 100% / 0.06)",
        backgroundColor: "hsl(var(--background) / 0.7)",
      }}
    >
      <div className="mx-auto flex h-14 max-w-6xl items-center justify-between gap-4 px-4">
        <div className="flex items-center gap-2.5">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg border bg-card"
               style={{ borderColor: "hsl(var(--primary) / 0.4)" }}>
            <Cpu className="h-4 w-4 text-[hsl(var(--primary))]" />
          </div>
          <div className="flex items-baseline gap-1.5">
            <span className="text-sm font-semibold tracking-tight">GPU Recommender</span>
            <span className="hidden font-mono text-[10px] uppercase tracking-widest text-muted-foreground sm:inline">
              v0.1
            </span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <StatusPill />
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}
