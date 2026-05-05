import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { AnimatePresence, motion } from "framer-motion";
import { toast } from "sonner";
import { AlertTriangle } from "lucide-react";

import { TopBar } from "@/components/TopBar";
import { SearchBox } from "@/components/SearchBox";
import { EmptyState } from "@/components/EmptyState";
import { SeedCard } from "@/components/SeedCard";
import { MetricStrip } from "@/components/MetricStrip";
import { RecommendationsGrid } from "@/components/RecommendationsGrid";
import { Footer } from "@/components/Footer";
import { ApiError, api, type Recommendation, type SearchResult } from "@/lib/api";

interface Seed {
  asin: string;
  title: string;
}

const Index = () => {
  const [seed, setSeed] = useState<Seed | null>(null);
  const [topK, setTopK] = useState(10);
  const [debouncedTopK, setDebouncedTopK] = useState(10);
  const [recs, setRecs] = useState<Recommendation[]>([]);
  const [loading, setLoading] = useState(false);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [latencyHistory, setLatencyHistory] = useState<number[]>([]);
  const [history, setHistory] = useState<Seed[]>([]);
  const [searchPrefill, setSearchPrefill] = useState<string | null>(null);
  const reqIdRef = useRef(0);
  const abortRef = useRef<AbortController | null>(null);

  // Debounce slider
  useEffect(() => {
    const t = setTimeout(() => setDebouncedTopK(topK), 250);
    return () => clearTimeout(t);
  }, [topK]);

  // Health check banner
  const { data: health } = useQuery({
    queryKey: ["health"],
    queryFn: ({ signal }) => api.health(signal),
    refetchInterval: 30_000,
    retry: 1,
    staleTime: 25_000,
  });

  // Fetch recs whenever seed or debouncedTopK changes
  useEffect(() => {
    if (!seed) return;
    const myId = ++reqIdRef.current;
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setLoading(true);

    api
      .recommend(seed.asin, debouncedTopK, ctrl.signal)
      .then((res) => {
        if (myId !== reqIdRef.current) return;
        setRecs(res.recommendations);
        setLatencyMs(res.inference_ms);
        setLatencyHistory((prev) => [...prev, res.inference_ms].slice(-10));
      })
      .catch((err: unknown) => {
        if ((err as Error).name === "AbortError") return;
        if (myId !== reqIdRef.current) return;
        if (err instanceof ApiError) {
          if (err.status === 404) toast.error("ASIN not found in catalog");
          else if (err.status === 503) toast.error("Model still loading on the server");
          else if (err.status === 500) toast.error("Inference failed on the server");
          else toast.error(err.message);
        } else {
          toast.error("Could not reach the recommender API");
        }
      })
      .finally(() => {
        if (myId === reqIdRef.current) setLoading(false);
      });
  }, [seed, debouncedTopK]);

  function pickSeed(s: Seed) {
    setSeed(s);
    setHistory((prev) => {
      const next = [s, ...prev.filter((p) => p.asin !== s.asin)];
      return next.slice(0, 5);
    });
  }

  function handleSearchSelect(r: SearchResult) {
    pickSeed({ asin: r.asin, title: r.title });
  }

  function handlePivot(r: Recommendation) {
    pickSeed({ asin: r.asin, title: r.title });
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function handleRerun() {
    if (!seed) return;
    setDebouncedTopK((k) => k); // no-op trigger; force refetch via state churn
    // Trigger by re-setting seed reference
    setSeed({ ...seed });
  }

  return (
    <div className="min-h-screen">
      <TopBar />

      {health && !health.model_loaded && (
        <div
          className="border-b bg-[hsl(var(--warning)/0.12)]"
          style={{ borderColor: "hsl(var(--warning) / 0.3)" }}
        >
          <div className="mx-auto flex max-w-6xl items-center gap-2 px-4 py-2.5 text-sm text-[hsl(var(--warning))]">
            <AlertTriangle className="h-4 w-4" />
            Model still loading on the server — try again shortly.
          </div>
        </div>
      )}

      <main className="mx-auto max-w-6xl px-4 pb-8 pt-12 sm:pt-16">
        {/* HERO */}
        <section className="mx-auto max-w-3xl text-center">
          <h1 className="text-4xl font-semibold tracking-tight sm:text-5xl">
            Find products. See what's similar.{" "}
            <span className="text-[hsl(var(--primary))]">In milliseconds.</span>
          </h1>
          <p className="mx-auto mt-4 max-w-xl text-base text-muted-foreground">
            GPU-accelerated nearest-neighbor search across 2M+ Amazon products.
          </p>

          <div className="mt-8">
            <SearchBox key={searchPrefill ?? "init"} onSelect={handleSearchSelect} />
          </div>
        </section>

        {/* CONTENT */}
        <div className="mt-12">
          <AnimatePresence mode="wait" initial={false}>
            {!seed ? (
              <motion.div
                key="empty"
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.25 }}
              >
                <EmptyState onExampleClick={(q) => setSearchPrefill(q + Date.now())} />
                {/* Note: SearchBox remounts on key change; we put the example query into the field via a small effect below */}
                <PrefillBridge prefill={searchPrefill} />
              </motion.div>
            ) : (
              <motion.div
                key="results"
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
                className="space-y-8"
              >
                {history.length > 1 && (
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-xs uppercase tracking-widest text-muted-foreground">
                      History
                    </span>
                    {history.map((h) => (
                      <button
                        key={h.asin}
                        onClick={() => pickSeed(h)}
                        className={`pressable chip max-w-[260px] truncate ${
                          h.asin === seed.asin
                            ? "border-[hsl(var(--primary)/0.5)] text-foreground"
                            : "text-muted-foreground hover:text-foreground"
                        }`}
                        title={h.title}
                      >
                        <span className="truncate">{h.title}</span>
                      </button>
                    ))}
                  </div>
                )}

                <div className="grid gap-4 lg:grid-cols-3">
                  <div className="lg:col-span-2">
                    <SeedCard
                      title={seed.title}
                      asin={seed.asin}
                      topK={topK}
                      onTopKChange={setTopK}
                      onRerun={handleRerun}
                      loading={loading}
                    />
                  </div>
                  <div>
                    <MetricStrip ms={latencyMs} history={latencyHistory} />
                  </div>
                </div>

                <RecommendationsGrid
                  items={recs}
                  loading={loading}
                  topK={debouncedTopK}
                  onPivot={handlePivot}
                />
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </main>

      <Footer />
    </div>
  );
};

/**
 * Tiny helper that focuses the search input and fills it when a chip is clicked.
 */
function PrefillBridge({ prefill }: { prefill: string | null }) {
  useEffect(() => {
    if (!prefill) return;
    const input = document.querySelector<HTMLInputElement>('input[aria-label="Search products"]');
    if (!input) return;
    const cleaned = prefill.replace(/\d+$/, "");
    const setter = Object.getOwnPropertyDescriptor(
      window.HTMLInputElement.prototype,
      "value",
    )?.set;
    setter?.call(input, cleaned);
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.focus();
  }, [prefill]);
  return null;
}

export default Index;
