import { useState } from "react";
import { Check, Copy } from "lucide-react";
import { cn } from "@/lib/utils";

export function AsinChip({ asin, className }: { asin: string; className?: string }) {
  const [copied, setCopied] = useState(false);

  async function copy(e: React.MouseEvent) {
    e.stopPropagation();
    try {
      await navigator.clipboard.writeText(asin);
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } catch {
      /* ignore */
    }
  }

  return (
    <button
      type="button"
      onClick={copy}
      title="Copy ASIN"
      className={cn(
        "chip font-mono text-[11px] text-muted-foreground hover:text-foreground transition-colors",
        className,
      )}
    >
      <span>{asin}</span>
      {copied ? (
        <Check className="h-3 w-3 text-[hsl(var(--success))]" />
      ) : (
        <Copy className="h-3 w-3 opacity-60" />
      )}
    </button>
  );
}
