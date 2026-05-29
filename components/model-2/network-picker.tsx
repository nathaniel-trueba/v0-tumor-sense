"use client";

import { useState } from "react";
import { ChevronDown } from "lucide-react";
import { NETWORKS, type Network } from "@/lib/breast-cancer/networks";

interface NetworkPickerProps {
  network: Network;
  onChange: (network: Network) => void;
}

const FAMILY_LABEL: Record<string, string> = {
  cnn: "CNN",
  resnet: "ResNet",
  vgg: "VGG",
  mobilenet: "MobileNet",
  vit: "Transformer",
  unet: "Segmentation",
};

export function NetworkPicker({ network, onChange }: NetworkPickerProps) {
  const [open, setOpen] = useState(false);

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setOpen((p) => !p)}
        className="flex items-center gap-3 border border-foreground/15 rounded-full px-5 py-3 text-sm hover:border-foreground/40 transition-colors"
      >
        <span className="font-mono text-xs uppercase tracking-wider text-muted-foreground">
          architecture
        </span>
        <span className="font-display text-lg">{network.name}</span>
        <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground border border-foreground/15 px-1.5 py-0.5 rounded">
          {FAMILY_LABEL[network.family] ?? network.family}
        </span>
        <ChevronDown
          className={`w-4 h-4 transition-transform ${open ? "rotate-180" : ""}`}
        />
      </button>
      {open && (
        <div className="absolute right-0 top-full mt-2 z-40 w-[420px] max-w-[90vw] bg-background border border-foreground/15 rounded-2xl shadow-lg overflow-hidden">
          {NETWORKS.map((n) => {
            const isActive = n.id === network.id;
            return (
              <button
                key={n.id}
                type="button"
                onClick={() => {
                  onChange(n);
                  setOpen(false);
                }}
                className={`w-full text-left px-5 py-4 border-b border-foreground/10 last:border-b-0 transition-colors ${
                  isActive ? "bg-foreground/[0.04]" : "hover:bg-foreground/[0.02]"
                }`}
              >
                <div className="flex items-baseline justify-between gap-3 mb-1">
                  <span className="font-display text-base">{n.name}</span>
                  <span className="text-xs font-mono text-muted-foreground">
                    {(n.accuracy * 100).toFixed(1)}% acc
                  </span>
                </div>
                <div className="flex items-center gap-3 text-xs font-mono text-muted-foreground mb-1">
                  <span>{FAMILY_LABEL[n.family]}</span>
                  <span>·</span>
                  <span>{n.paramsM.toFixed(2)} M params</span>
                  <span>·</span>
                  <span>{n.flopsG.toFixed(3)} GFLOPs</span>
                  <span>·</span>
                  <span>{n.latencyMs.toFixed(1)} ms</span>
                </div>
                <p className="text-xs text-muted-foreground leading-relaxed">{n.description}</p>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
