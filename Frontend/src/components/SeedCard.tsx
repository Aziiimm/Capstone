import { Cpu, RotateCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { AsinChip } from "./AsinChip";

interface Props {
  title: string;
  asin: string;
  topK: number;
  onTopKChange: (n: number) => void;
  onRerun: () => void;
  loading: boolean;
}

export function SeedCard({ title, asin, topK, onTopKChange, onRerun, loading }: Props) {
  return (
    <div className="card-surface relative overflow-hidden p-6">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 text-[11px] font-medium uppercase tracking-widest text-[hsl(var(--primary))]">
            <Cpu className="h-3.5 w-3.5" />
            Seed product
          </div>
          <h2 className="mt-2 text-xl font-semibold leading-snug sm:text-2xl">
            {title}
          </h2>
          <div className="mt-3">
            <AsinChip asin={asin} />
          </div>
        </div>

        {loading && (
          <div className="shrink-0 rounded-full border bg-card/70 p-2 animate-pulse-soft"
               style={{ borderColor: "hsl(var(--primary) / 0.3)" }}>
            <Cpu className="h-4 w-4 text-[hsl(var(--primary))]" />
          </div>
        )}
      </div>

      <div className="mt-6 flex flex-wrap items-center gap-x-6 gap-y-4">
        <div className="flex min-w-[260px] flex-1 items-center gap-4">
          <label className="shrink-0 text-xs uppercase tracking-widest text-muted-foreground">
            Top K
          </label>
          <Slider
            value={[topK]}
            min={1}
            max={100}
            step={1}
            onValueChange={(v) => onTopKChange(v[0])}
            className="flex-1"
          />
          <span className="w-10 shrink-0 text-right font-mono text-sm tabular-nums">
            {topK}
          </span>
        </div>

        <Button
          onClick={onRerun}
          disabled={loading}
          className="pressable bg-[hsl(var(--primary))] text-[hsl(var(--primary-foreground))] hover:bg-[hsl(var(--primary)/0.9)]"
        >
          <RotateCw className={loading ? "h-4 w-4 animate-spin" : "h-4 w-4"} />
          Get Recommendations
        </Button>
      </div>
    </div>
  );
}
