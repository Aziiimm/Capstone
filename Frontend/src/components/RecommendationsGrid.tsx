import { motion } from "framer-motion";
import { ArrowRight } from "lucide-react";
import type { Recommendation } from "@/lib/api";
import { AsinChip } from "./AsinChip";

interface Props {
  items: Recommendation[];
  loading: boolean;
  topK: number;
  onPivot: (r: Recommendation) => void;
}

export function RecommendationsGrid({ items, loading, topK, onPivot }: Props) {
  return (
    <section>
      <div className="mb-4 flex items-baseline justify-between">
        <h3 className="text-lg font-semibold tracking-tight">
          Top {topK} similar products
        </h3>
        <span className="font-mono text-xs text-muted-foreground">
          {items.length > 0 ? `${items.length} results` : ""}
        </span>
      </div>

      {loading && items.length === 0 ? (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {Array.from({ length: Math.min(topK, 8) }).map((_, i) => (
            <div key={i} className="card-surface h-44 animate-pulse-soft" />
          ))}
        </div>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {items.map((r, i) => (
            <motion.article
              key={`${r.asin}-${i}`}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.25, delay: Math.min(i * 0.02, 0.2) }}
              className="card-surface card-hover flex flex-col p-4"
            >
              <div className="mb-3 flex items-center justify-between">
                <span className="font-mono text-[11px] font-semibold text-[hsl(var(--primary))]">
                  #{i + 1}
                </span>
                <AsinChip asin={r.asin} />
              </div>
              <p className="line-clamp-3 flex-1 text-sm leading-snug text-foreground">
                {r.title}
              </p>
              <button
                onClick={() => onPivot(r)}
                className="pressable mt-4 inline-flex items-center gap-1.5 self-start rounded-md text-xs font-medium text-[hsl(var(--primary))] hover:gap-2 transition-all"
              >
                Use as seed <ArrowRight className="h-3.5 w-3.5" />
              </button>
            </motion.article>
          ))}
        </div>
      )}
    </section>
  );
}
