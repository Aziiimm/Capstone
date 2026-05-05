import { useState } from "react";
import { ChevronDown, Database, Layers, Zap } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { formatCount } from "@/lib/format";

interface Props {
  onExampleClick: (q: string) => void;
}

const EXAMPLES = [
  "binoculars",
  "cordless drill",
  "yoga mat",
  "headlamp",
  "monitor stand",
  "tape measure",
];

const STEPS = [
  {
    n: 1,
    title: "Search the catalog",
    body: "Type a product name. Titles are matched against 2.17M Amazon items.",
  },
  {
    n: 2,
    title: "Pick a seed product",
    body: "Selecting a result locks in an ASIN as the query vector for the GPU model.",
  },
  {
    n: 3,
    title: "GPU computes neighbors",
    body: "cuML NearestNeighbors runs on GPU and returns the K most similar items in milliseconds.",
  },
];

export function EmptyState({ onExampleClick }: Props) {
  const [open, setOpen] = useState(false);
  const { data } = useQuery({
    queryKey: ["health"],
    queryFn: ({ signal }) => api.health(signal),
    staleTime: 25_000,
  });

  const catalog = data?.catalogue_size ? formatCount(data.catalogue_size) : "2.17M";

  const tiles = [
    { icon: Database, label: "Catalog size", value: catalog, sub: "products indexed" },
    { icon: Layers, label: "Categories", value: "3", sub: "Tools · Industrial · Sports" },
    { icon: Zap, label: "Avg warm latency", value: "~90", sub: "ms per query", unit: "ms" },
  ];

  return (
    <div className="space-y-8">
      <div className="grid gap-4 sm:grid-cols-3">
        {tiles.map((t) => (
          <div key={t.label} className="card-surface p-5">
            <div className="flex items-center gap-2 text-[11px] font-medium uppercase tracking-widest text-muted-foreground">
              <t.icon className="h-3.5 w-3.5" />
              {t.label}
            </div>
            <div className="mt-2 font-mono text-2xl font-semibold tabular-nums">
              {t.value}
            </div>
            <div className="mt-1 text-xs text-muted-foreground">{t.sub}</div>
          </div>
        ))}
      </div>

      <div>
        <div className="mb-3 text-xs uppercase tracking-widest text-muted-foreground">
          Try an example query
        </div>
        <div className="flex flex-wrap gap-2">
          {EXAMPLES.map((q) => (
            <button
              key={q}
              onClick={() => onExampleClick(q)}
              className="pressable chip text-foreground/80 hover:border-[hsl(var(--primary)/0.5)] hover:text-foreground transition-colors"
            >
              {q}
            </button>
          ))}
        </div>
      </div>

      <div className="card-surface overflow-hidden">
        <button
          onClick={() => setOpen((o) => !o)}
          className="flex w-full items-center justify-between p-5 text-left"
        >
          <div>
            <div className="text-sm font-semibold">How it works</div>
            <div className="text-xs text-muted-foreground">
              cuML NearestNeighbors on GPU · 3 steps
            </div>
          </div>
          <ChevronDown className={`h-4 w-4 transition-transform ${open ? "rotate-180" : ""}`} />
        </button>
        <AnimatePresence initial={false}>
          {open && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.25 }}
              className="overflow-hidden"
            >
              <div className="grid gap-4 border-t p-5 sm:grid-cols-3"
                   style={{ borderColor: "hsl(0 0% 100% / 0.06)" }}>
                {STEPS.map((s) => (
                  <div key={s.n} className="rounded-xl border p-4"
                       style={{ borderColor: "hsl(0 0% 100% / 0.06)" }}>
                    <div className="flex items-center gap-2 font-mono text-xs text-[hsl(var(--primary))]">
                      step {String(s.n).padStart(2, "0")}
                    </div>
                    <div className="mt-1.5 text-sm font-medium">{s.title}</div>
                    <p className="mt-1.5 text-xs leading-relaxed text-muted-foreground">{s.body}</p>
                  </div>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
