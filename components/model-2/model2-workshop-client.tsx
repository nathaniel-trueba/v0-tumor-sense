"use client";

import dynamic from "next/dynamic";

// Client-only loader for the workshop. We mount it with ssr:false to avoid
// hydration mismatches: the Cursor MCP browser overlay injects data-cursor-ref
// attributes into the DOM before React hydrates, which otherwise trips React
// 19's strict hydration diff. The page shell (nav + footer) still SSRs.
const Model2Workshop = dynamic(
  () => import("./model2-workshop").then((m) => m.Model2Workshop),
  {
    ssr: false,
    loading: () => <WorkshopSkeleton />,
  }
);

export function Model2WorkshopClient() {
  return <Model2Workshop />;
}

function WorkshopSkeleton() {
  return (
    <div className="space-y-16 animate-pulse">
      <div className="space-y-4">
        <div className="h-3 w-48 bg-foreground/5 rounded-full" />
        <div className="h-14 w-2/3 bg-foreground/5 rounded-md" />
        <div className="h-14 w-1/2 bg-foreground/5 rounded-md" />
        <div className="h-4 w-full max-w-3xl bg-foreground/5 rounded" />
      </div>
      <div className="h-72 w-full bg-foreground/5 rounded-md" />
      <div className="grid grid-cols-6 sm:grid-cols-9 gap-2">
        {Array.from({ length: 18 }).map((_, i) => (
          <div key={i} className="aspect-square bg-foreground/5 rounded-md" />
        ))}
      </div>
      <div className="h-96 w-full bg-foreground/5 rounded-md" />
    </div>
  );
}
