"use client";

import { useState } from "react";
import Image from "next/image";
import { COUNTS, IMAGE_CATALOG, type PatchImage } from "@/lib/breast-cancer/image-dataset";

const CANCER = "#EC4899";
const NO_CANCER = "#0F6BFF";

type Filter = "all" | "cancer" | "no_cancer";

interface ImageGalleryProps {
  selected: PatchImage;
  onSelect: (image: PatchImage) => void;
}

export function ImageGallery({ selected, onSelect }: ImageGalleryProps) {
  const [filter, setFilter] = useState<Filter>("all");
  const filtered =
    filter === "all" ? IMAGE_CATALOG : IMAGE_CATALOG.filter((i) => i.label === filter);

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div className="flex items-center gap-2">
          {(["all", "cancer", "no_cancer"] as Filter[]).map((f) => (
            <button
              key={f}
              type="button"
              onClick={() => setFilter(f)}
              className={`text-xs font-mono px-3 py-1.5 rounded-md border transition-colors ${
                filter === f
                  ? "bg-foreground text-background border-foreground"
                  : "border-foreground/15 hover:border-foreground/30"
              }`}
            >
              {f === "all"
                ? `all · ${COUNTS.total}`
                : f === "cancer"
                ? `cancer · ${COUNTS.cancer}`
                : `no cancer · ${COUNTS.no_cancer}`}
            </button>
          ))}
        </div>
        <span className="text-xs font-mono text-muted-foreground">
          50×50 IDC patches · click to set as input
        </span>
      </div>

      <div className="grid grid-cols-6 sm:grid-cols-8 lg:grid-cols-9 gap-2">
        {filtered.map((img) => {
          const isSelected = img.id === selected.id;
          const color = img.label === "cancer" ? CANCER : NO_CANCER;
          return (
            <button
              key={img.id}
              type="button"
              onClick={() => onSelect(img)}
              className={`group relative aspect-square rounded-md overflow-hidden border-2 transition-all ${
                isSelected
                  ? "ring-2 ring-offset-2 ring-offset-background scale-105"
                  : "border-transparent hover:scale-105"
              }`}
              style={{
                borderColor: isSelected ? color : "transparent",
                boxShadow: isSelected ? `0 0 0 1px ${color}` : undefined,
              }}
              aria-label={`${img.id} · ${img.label}`}
            >
              <Image
                src={img.url}
                alt={img.id}
                fill
                sizes="80px"
                unoptimized
                className="object-cover [image-rendering:pixelated]"
              />
              <span
                className="absolute bottom-0 left-0 right-0 h-1 transition-opacity"
                style={{
                  background: color,
                  opacity: isSelected ? 1 : 0.55,
                }}
              />
              <span
                className="absolute top-0.5 left-0.5 text-[8px] font-mono px-1 rounded-sm bg-background/80 backdrop-blur opacity-0 group-hover:opacity-100 transition-opacity"
              >
                {img.id.replace("grid_", "#")}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

export function SelectedImagePreview({ image }: { image: PatchImage }) {
  const color = image.label === "cancer" ? CANCER : NO_CANCER;
  return (
    <div className="border border-foreground/10 bg-background p-5 flex flex-col sm:flex-row gap-5">
      <div
        className="relative w-32 h-32 sm:w-40 sm:h-40 border-2 rounded-md overflow-hidden shrink-0"
        style={{ borderColor: color }}
      >
        <Image
          src={image.url}
          alt={image.id}
          fill
          sizes="160px"
          unoptimized
          className="object-cover [image-rendering:pixelated]"
        />
      </div>
      <div className="space-y-2 flex-1 min-w-0">
        <div className="flex items-baseline justify-between gap-3 flex-wrap">
          <div className="font-mono text-xs text-muted-foreground">{image.id}</div>
          <span
            className="text-[10px] font-mono uppercase tracking-wider px-2 py-0.5 rounded-full"
            style={{ background: `${color}1f`, color }}
          >
            ground truth · {image.label.replace("_", " ")}
          </span>
        </div>
        <div className="font-display text-2xl">Patient {image.patientId}</div>
        <div className="grid grid-cols-2 gap-x-4 gap-y-1 font-mono text-xs text-muted-foreground">
          <span>x: {image.x}px</span>
          <span>y: {image.y}px</span>
          <span>size: 50×50</span>
          <span>channels: RGB</span>
        </div>
        <div className="text-xs text-muted-foreground/80 leading-relaxed pt-1">
          IDC-{image.label === "cancer" ? "positive" : "negative"} patch · sampled from the
          Kaggle Invasive Ductal Carcinoma dataset.
        </div>
      </div>
    </div>
  );
}
