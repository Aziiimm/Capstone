import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { formatCount } from "@/lib/format";
import { cn } from "@/lib/utils";

export function StatusPill() {
  const { data, isError, isLoading } = useQuery({
    queryKey: ["health"],
    queryFn: ({ signal }) => api.health(signal),
    refetchInterval: 30_000,
    retry: 1,
    staleTime: 25_000,
  });

  let tone: "good" | "warn" | "bad" = "warn";
  let label = "Connecting…";
  let detail = "";

  if (isError) {
    tone = "bad";
    label = "Offline";
  } else if (data) {
    if (data.model_loaded) {
      tone = "good";
      label = "Model Ready";
      detail = `${formatCount(data.catalogue_size)} items`;
    } else {
      tone = "warn";
      label = "Loading model…";
    }
  } else if (isLoading) {
    tone = "warn";
    label = "Connecting…";
  }

  return (
    <div
      className="inline-flex items-center gap-2 rounded-full border bg-card/60 px-3 py-1.5 text-xs backdrop-blur-sm"
      style={{ borderColor: "hsl(0 0% 100% / 0.08)" }}
    >
      <span
        className={cn(
          "h-2 w-2 rounded-full",
          tone === "good" && "bg-[hsl(var(--success))] glow-dot",
          tone === "warn" && "bg-[hsl(var(--warning))]",
          tone === "bad" && "bg-[hsl(var(--destructive))]",
        )}
      />
      <span className="font-medium text-foreground">{label}</span>
      {detail && (
        <span className="font-mono text-muted-foreground">· {detail}</span>
      )}
    </div>
  );
}
