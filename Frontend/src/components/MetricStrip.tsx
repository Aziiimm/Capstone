import { latencyTone } from "@/lib/format";
import { cn } from "@/lib/utils";

interface Props {
  ms: number | null;
  history: number[];
}

export function MetricStrip({ ms, history }: Props) {
  const tone = ms != null ? latencyTone(ms) : null;
  const toneClass =
    tone === "good"
      ? "text-[hsl(var(--success))]"
      : tone === "warn"
      ? "text-[hsl(var(--warning))]"
      : tone === "bad"
      ? "text-[hsl(var(--destructive))]"
      : "text-muted-foreground";

  return (
    <div className="card-surface flex h-full flex-col justify-between p-5">
      <div>
        <div className="text-[11px] font-medium uppercase tracking-widest text-muted-foreground">
          GPU inference time
        </div>
        <div className={cn("mt-2 font-mono text-4xl font-semibold tabular-nums", toneClass)}>
          {ms != null ? ms.toFixed(1) : "—"}
          <span className="ml-1 text-base text-muted-foreground">ms</span>
        </div>
      </div>

      <Sparkline values={history} />

      <p className="mt-3 text-[11px] leading-relaxed text-muted-foreground">
        First request after model load is slower (~350ms). Subsequent queries run on warm GPU.
      </p>
    </div>
  );
}

function Sparkline({ values }: { values: number[] }) {
  const W = 220;
  const H = 40;
  if (values.length < 2) {
    return (
      <div className="mt-3 h-[40px] rounded-md border border-dashed text-center text-[10px] leading-[40px] text-muted-foreground"
           style={{ borderColor: "hsl(0 0% 100% / 0.08)" }}>
        collecting samples…
      </div>
    );
  }
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = Math.max(1, max - min);
  const step = W / (values.length - 1);
  const points = values
    .map((v, i) => {
      const x = i * step;
      const y = H - ((v - min) / span) * (H - 4) - 2;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  const area = `0,${H} ${points} ${W},${H}`;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="mt-3 h-10 w-full" preserveAspectRatio="none">
      <defs>
        <linearGradient id="spark" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor="hsl(187 92% 55%)" stopOpacity="0.45" />
          <stop offset="100%" stopColor="hsl(187 92% 55%)" stopOpacity="0" />
        </linearGradient>
      </defs>
      <polygon points={area} fill="url(#spark)" />
      <polyline
        points={points}
        fill="none"
        stroke="hsl(187 92% 55%)"
        strokeWidth="1.5"
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  );
}
