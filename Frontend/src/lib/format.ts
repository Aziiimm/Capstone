export function formatCount(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return String(n);
}

export function latencyTone(ms: number): "good" | "warn" | "bad" {
  if (ms < 150) return "good";
  if (ms <= 500) return "warn";
  return "bad";
}
