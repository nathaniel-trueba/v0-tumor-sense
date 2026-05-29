"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { Lightbulb, ScanLine, ImageDown, Cpu, Layers, ArrowUpRight, Sliders } from "lucide-react";
import { getDefaultImage, type PatchImage } from "@/lib/breast-cancer/image-dataset";
import { NETWORK_INDEX, NETWORKS, type Network } from "@/lib/breast-cancer/networks";
import { NetworkPicker } from "./network-picker";
import { Leaderboard } from "./leaderboard";
import { ImageGallery, SelectedImagePreview } from "./image-gallery";
import { Architecture3D } from "./architecture-3d";
import { ForwardPassViz } from "./forward-pass-viz";
import { SaliencyView } from "./saliency-view";
import { EmbeddingExplorer } from "./embedding-explorer";
import { EigenBreasts } from "./eigen-breasts";
import { SegmentationView } from "./segmentation-view";
import { NetworkStats } from "./network-stats";

type Mode = "classification" | "segmentation";

export function Model2Workshop() {
  const [image, setImage] = useState<PatchImage>(() => getDefaultImage());
  const [network, setNetwork] = useState<Network>(() => NETWORK_INDEX.resnet18);
  const [mode, setMode] = useState<Mode>("classification");
  const [activeLayerIndex, setActiveLayerIndex] = useState<number>(network.layers.length - 1);

  // Reset layer index when the network changes.
  useEffect(() => {
    setActiveLayerIndex(network.layers.length - 1);
  }, [network.id, network.layers.length]);

  // When switching into segmentation mode, prefer the U-Net.
  useEffect(() => {
    if (mode === "segmentation" && network.family !== "unet") {
      setNetwork(NETWORK_INDEX.unet);
    }
  }, [mode, network.family]);

  const classificationNetworks = useMemo(
    () => NETWORKS.filter((n) => n.family !== "unet") as Network[],
    []
  );

  return (
    <div className="space-y-16 lg:space-y-24">
      {/* ── Hero ─────────────────────────────────────────────────────────── */}
      <section className="space-y-6">
        <div className="flex items-center justify-between gap-4 flex-wrap">
          <span className="inline-flex items-center gap-3 text-sm font-mono text-muted-foreground">
            <span className="w-8 h-px bg-foreground/30" />
            Image model workshop · prototype
          </span>
          <Link
            href="/model"
            className="group inline-flex items-center gap-2 rounded-full border border-foreground/15 hover:border-foreground/40 bg-background/40 px-4 py-1.5 text-xs font-mono text-muted-foreground hover:text-foreground transition-colors"
          >
            <Sliders className="w-3.5 h-3.5" />
            <span className="uppercase tracking-wider">Back to SVM workshop</span>
            <ArrowUpRight className="w-3.5 h-3.5 -translate-x-0.5 group-hover:translate-x-0 transition-transform" />
          </Link>
        </div>
        <div className="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
          <h1 className="text-4xl lg:text-6xl font-display tracking-tight max-w-3xl">
            Watch the network <br />
            <span className="text-muted-foreground">see the tissue.</span>
          </h1>
          <NetworkPicker network={network} onChange={setNetwork} />
        </div>
        <p className="text-lg text-muted-foreground max-w-3xl leading-relaxed">
          Pick a histopathology patch, pick an architecture, and step through the
          forward pass. Hover any layer in the 3D stack — or scrub the strip below —
          to see what the network is computing on the way to its decision.
        </p>
      </section>

      {/* ── Brainstorm callout (temporarily hidden) ──────────────────────── */}
      {/* <BrainstormCallout /> */}

      {/* ── Leaderboard ──────────────────────────────────────────────────── */}
      <Leaderboard selectedId={network.id} onSelect={setNetwork} />

      {/* ── Image picker ─────────────────────────────────────────────────── */}
      <section className="space-y-6">
        <div className="flex items-baseline justify-between gap-4 flex-wrap">
          <h2 className="font-display text-2xl lg:text-3xl tracking-tight">Input gallery</h2>
          <span className="text-xs font-mono text-muted-foreground">
            36 IDC histology patches · 50×50 RGB · class-balanced
          </span>
        </div>
        <ImageGallery selected={image} onSelect={setImage} />
        <SelectedImagePreview image={image} />
      </section>

      {/* ── Mode tabs (classification / segmentation) ────────────────────── */}
      <section className="space-y-8">
        <div className="grid sm:grid-cols-2 gap-px bg-foreground/10 border border-foreground/10">
          {([
            {
              id: "classification" as Mode,
              label: "Classification",
              sub: "predict patch label · benign vs malignant",
              icon: Cpu,
            },
            {
              id: "segmentation" as Mode,
              label: "Segmentation",
              sub: "per-pixel tumor mask via U-Net",
              icon: ScanLine,
            },
          ]).map((tab) => {
            const Icon = tab.icon;
            const isActive = tab.id === mode;
            return (
              <button
                key={tab.id}
                type="button"
                onClick={() => setMode(tab.id)}
                className={`p-6 text-left transition-colors relative ${
                  isActive
                    ? "bg-foreground text-background"
                    : "bg-background hover:bg-foreground/[0.03]"
                }`}
              >
                <div className="flex items-center gap-3 mb-3">
                  <Icon
                    className={`w-4 h-4 ${
                      isActive ? "text-background" : "text-foreground/70"
                    }`}
                  />
                  <span
                    className={`font-mono text-[11px] uppercase tracking-wider ${
                      isActive ? "text-background/70" : "text-muted-foreground"
                    }`}
                  >
                    task
                  </span>
                </div>
                <div className="font-display text-2xl">{tab.label}</div>
                <div
                  className={`text-xs font-mono mt-1 ${
                    isActive ? "text-background/70" : "text-muted-foreground"
                  }`}
                >
                  {tab.sub}
                </div>
                {isActive && (
                  <span className="absolute -bottom-px left-0 right-0 h-0.5 bg-background" />
                )}
              </button>
            );
          })}
        </div>

        {mode === "classification" ? (
          <ClassificationPanels
            network={network}
            image={image}
            activeLayerIndex={activeLayerIndex}
            onActiveLayerChange={setActiveLayerIndex}
            networksForFamily={classificationNetworks}
          />
        ) : (
          <SegmentationPanels image={image} />
        )}
      </section>

      {/* ── Network stats ────────────────────────────────────────────────── */}
      <section className="space-y-6">
        <div className="flex items-baseline justify-between gap-4 flex-wrap">
          <h2 className="font-display text-3xl lg:text-5xl tracking-tight">Model-specific stats</h2>
          <span className="text-sm font-mono text-muted-foreground">
            {network.name} · {network.layers.length} layers · {network.paramsM.toFixed(2)} M params
          </span>
        </div>
        <NetworkStats network={network} />
      </section>

      {/* ── Brainstormed extras: clearly labelled section ───────────────── */}
      {mode === "classification" && (
        <section className="space-y-10">
          <div>
            <span className="inline-flex items-center gap-3 text-sm font-mono text-muted-foreground mb-4">
              <span className="w-8 h-px bg-foreground/30" />
              Extras — brainstormed XAI panels
            </span>
            <h2 className="text-3xl lg:text-5xl font-display tracking-tight max-w-3xl">
              Where the model looks. <br />
              <span className="text-muted-foreground">How its space curves.</span>
            </h2>
          </div>
          <SaliencyView network={network} image={image} />
          <div className="border-t border-foreground/10 pt-10">
            <EmbeddingExplorer network={network} selected={image} onSelect={setImage} />
          </div>
          <div className="border-t border-foreground/10 pt-10">
            <EigenBreasts image={image} />
          </div>
        </section>
      )}
    </div>
  );
}

function ClassificationPanels({
  network,
  image,
  activeLayerIndex,
  onActiveLayerChange,
  networksForFamily,
}: {
  network: Network;
  image: PatchImage;
  activeLayerIndex: number;
  onActiveLayerChange: (i: number) => void;
  networksForFamily: Network[];
}) {
  return (
    <div className="space-y-10">
      {/* 3D architecture */}
      <div className="space-y-5">
        <div className="flex items-baseline justify-between gap-3 flex-wrap">
          <div className="flex items-center gap-3">
            <Layers className="w-4 h-4 text-muted-foreground" />
            <h3 className="font-display text-2xl lg:text-3xl tracking-tight">
              Architecture in 3D
            </h3>
          </div>
          <span className="text-xs font-mono text-muted-foreground">
            slab area = spatial resolution · slab depth = channel count
          </span>
        </div>
        <Architecture3D
          network={network}
          activeIndex={activeLayerIndex}
          onActiveChange={onActiveLayerChange}
        />
      </div>

      {/* Forward pass */}
      <div className="space-y-5 pt-2">
        <div className="flex items-baseline justify-between gap-3 flex-wrap">
          <div className="flex items-center gap-3">
            <ImageDown className="w-4 h-4 text-muted-foreground" />
            <h3 className="font-display text-2xl lg:text-3xl tracking-tight">
              Forward pass · feature maps
            </h3>
          </div>
          <span className="text-xs font-mono text-muted-foreground">
            real conv filters applied to the selected patch
          </span>
        </div>
        <ForwardPassViz network={network} image={image} />
      </div>

      {/* Subtle note about which networks are in scope */}
      <div className="text-xs font-mono text-muted-foreground border-t border-foreground/10 pt-4">
        Classification scope · {networksForFamily.length} networks (CNN, ResNet, VGG, MobileNet, ViT).
        The U-Net is reserved for the segmentation tab above.
      </div>
    </div>
  );
}

function SegmentationPanels({ image }: { image: PatchImage }) {
  return (
    <div className="space-y-10">
      <SegmentationView image={image} />
      <p className="text-xs font-mono text-muted-foreground border-t border-foreground/10 pt-4">
        Mock U-Net output. Real masks would come from a trained segmentation head
        on the BACH / BreastPathQ datasets — wiring is left for the backend.
      </p>
    </div>
  );
}

function BrainstormCallout() {
  return (
    <section className="border border-foreground/10 rounded-xl p-6 lg:p-8 bg-foreground/[0.02]">
      <div className="flex items-start gap-4">
        <div className="shrink-0 w-10 h-10 rounded-full bg-foreground text-background flex items-center justify-center">
          <Lightbulb className="w-4 h-4" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="text-[11px] font-mono uppercase tracking-wider text-muted-foreground">
            Brainstormed extras
          </div>
          <h3 className="font-display text-xl lg:text-2xl mt-1">
            Two additional XAI ideas added to this page
          </h3>
          <div className="grid sm:grid-cols-2 gap-6 mt-5">
            <BrainstormItem
              title="Grad-CAM saliency overlay"
              desc="A live heatmap of where the chosen network was 'looking' when it scored the patch. Toggle between pixel and coarse 10×10 resolutions to mimic the activation-map upscaling that real Grad-CAM does."
            />
            <BrainstormItem
              title="Embedding-space explorer"
              desc="Project every patch's penultimate-layer activation into 2D (UMAP / PCA). Hover any point to preview the patch, click to set it as the input — and watch the malignant cluster pull away from the benign one as you switch networks."
            />
            <BrainstormItem
              title="Eigen-breasts (PCA)"
              desc="Real PCA over all 36 patches. The top components surface as bipolar 'pink ↔ blue' axes — a representation-learning lens that pairs naturally with the network-derived embeddings."
            />
            <BrainstormItem
              title="Segmentation alt-view"
              desc="An optional U-Net tab that produces per-pixel tumor masks with Dice / IoU readouts. Easy to remove or extend later."
            />
          </div>
        </div>
      </div>
    </section>
  );
}

function BrainstormItem({ title, desc }: { title: string; desc: string }) {
  return (
    <div>
      <div className="font-display text-base">{title}</div>
      <p className="text-sm text-muted-foreground mt-1 leading-relaxed">{desc}</p>
    </div>
  );
}
