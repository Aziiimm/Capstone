export function Footer() {
  return (
    <footer
      className="mt-20 border-t py-8"
      style={{ borderColor: "hsl(0 0% 100% / 0.06)" }}
    >
      <div className="mx-auto flex max-w-6xl flex-col items-center justify-between gap-3 px-4 text-xs text-muted-foreground sm:flex-row">
        <span>Capstone project — GPU vs CPU collaborative filtering</span>
        <div className="flex items-center gap-5">
          <a href="#" className="hover:text-foreground transition-colors">GitHub</a>
          <a href="#" className="hover:text-foreground transition-colors">Docs</a>
        </div>
      </div>
    </footer>
  );
}
