"use client";

import { Suspense, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Group, Mesh, MeshStandardMaterial, Object3D } from "three";
import type { Layer, Network } from "@/lib/breast-cancer/networks";

const LAYER_COLORS: Record<string, string> = {
  input: "#0F6BFF",
  conv: "#EC4899",
  pool: "#9CA3AF",
  residual: "#7C3AED",
  depthwise: "#F59E0B",
  bn: "#10B981",
  relu: "#3B82F6",
  flatten: "#475569",
  fc: "#111111",
  softmax: "#EC4899",
};

interface LayerLayout {
  layer: Layer;
  index: number;
  z: number; // axial position along the network
  sizeX: number;
  sizeY: number;
  sizeZ: number; // depth (channels)
  color: string;
}

function layoutLayers(network: Network): LayerLayout[] {
  // Normalise spatial dims (50 -> 1.4 units) and channels (max -> 1.6 units).
  const maxSpatial = Math.max(...network.layers.map((l) => l.shape[0]));
  const maxChan = Math.max(...network.layers.map((l) => l.shape[2]));

  let zCursor = 0;
  const spacing = 0.35;
  return network.layers.map((layer, index) => {
    const spatial = Math.max(0.18, (layer.shape[0] / maxSpatial) * 1.5);
    const depth = Math.max(0.12, (layer.shape[2] / maxChan) * 1.6);
    const layout: LayerLayout = {
      layer,
      index,
      sizeX: spatial,
      sizeY: spatial,
      sizeZ: depth,
      z: zCursor + depth / 2,
      color: LAYER_COLORS[layer.kind] ?? "#444",
    };
    zCursor += depth + spacing;
    return layout;
  });
}

interface LayerMeshProps {
  layout: LayerLayout;
  active: boolean;
  hovered: boolean;
  onHover: (index: number | null) => void;
  onClick: (index: number) => void;
  totalLength: number;
}

function LayerSlab({ layout, active, hovered, onHover, onClick, totalLength }: LayerMeshProps) {
  const meshRef = useRef<Mesh>(null);
  const baseY = 0;
  const z = layout.z - totalLength / 2;
  // Stagger residual + depthwise vertically for a little visual rhythm.
  const yOffset =
    layout.layer.kind === "residual"
      ? 0.05
      : layout.layer.kind === "depthwise"
      ? -0.05
      : 0;

  useFrame(({ clock }) => {
    if (!meshRef.current) return;
    const m = meshRef.current.material as MeshStandardMaterial;
    const target = active ? 1 : hovered ? 0.85 : 0.55;
    m.emissiveIntensity += (target - m.emissiveIntensity) * 0.18;
    // Soft pulse on the active layer.
    if (active) {
      meshRef.current.scale.x = 1 + Math.sin(clock.elapsedTime * 3) * 0.02;
      meshRef.current.scale.y = 1 + Math.sin(clock.elapsedTime * 3) * 0.02;
    } else {
      meshRef.current.scale.x += (1 - meshRef.current.scale.x) * 0.18;
      meshRef.current.scale.y += (1 - meshRef.current.scale.y) * 0.18;
    }
  });

  return (
    <group position={[0, baseY + yOffset, z]}>
      <mesh
        ref={meshRef}
        onPointerOver={(e) => {
          e.stopPropagation();
          onHover(layout.index);
          document.body.style.cursor = "pointer";
        }}
        onPointerOut={() => {
          onHover(null);
          document.body.style.cursor = "";
        }}
        onClick={(e) => {
          e.stopPropagation();
          onClick(layout.index);
        }}
        castShadow
        receiveShadow
      >
        <boxGeometry args={[layout.sizeX, layout.sizeY, layout.sizeZ]} />
        <meshStandardMaterial
          color={layout.color}
          emissive={layout.color}
          emissiveIntensity={0.55}
          metalness={0.15}
          roughness={0.55}
          transparent
          opacity={0.92}
        />
      </mesh>
      {/* Subtle wireframe overlay */}
      <mesh>
        <boxGeometry args={[layout.sizeX * 1.002, layout.sizeY * 1.002, layout.sizeZ * 1.002]} />
        <meshBasicMaterial color="#111" wireframe transparent opacity={active || hovered ? 0.35 : 0.15} />
      </mesh>
    </group>
  );
}

function Scene({
  network,
  activeIndex,
  setActiveIndex,
  hoveredIndex,
  setHoveredIndex,
}: {
  network: Network;
  activeIndex: number;
  setActiveIndex: (i: number) => void;
  hoveredIndex: number | null;
  setHoveredIndex: (i: number | null) => void;
}) {
  const layouts = useMemo(() => layoutLayers(network), [network]);
  const totalLength =
    layouts[layouts.length - 1].z + layouts[layouts.length - 1].sizeZ / 2;
  const groupRef = useRef<Group>(null);
  const { mouse } = useThree();

  // Auto-rotate the network gently; let mouse parallax steer the rotation.
  useFrame(({ clock }) => {
    if (!groupRef.current) return;
    const baseRotY = clock.elapsedTime * 0.18 + mouse.x * 0.6;
    const baseRotX = -0.25 + mouse.y * -0.25;
    groupRef.current.rotation.y += (baseRotY - groupRef.current.rotation.y) * 0.08;
    groupRef.current.rotation.x += (baseRotX - groupRef.current.rotation.x) * 0.08;
  });

  return (
    <group ref={groupRef}>
      <ambientLight intensity={0.45} />
      <directionalLight position={[3, 5, 4]} intensity={1.2} castShadow />
      <directionalLight position={[-4, -2, 3]} intensity={0.35} color="#EC4899" />
      {layouts.map((l) => (
        <LayerSlab
          key={l.layer.id}
          layout={l}
          active={l.index === activeIndex}
          hovered={l.index === hoveredIndex}
          totalLength={totalLength}
          onHover={setHoveredIndex}
          onClick={setActiveIndex}
        />
      ))}
      {/* Connecting line through the centers */}
      <ConnectingLine layouts={layouts} totalLength={totalLength} />
    </group>
  );
}

function ConnectingLine({
  layouts,
  totalLength,
}: {
  layouts: LayerLayout[];
  totalLength: number;
}) {
  const ref = useRef<Object3D>(null);
  // Use a thin cylinder between each pair of layer centers.
  return (
    <group ref={ref}>
      {layouts.slice(0, -1).map((l, i) => {
        const next = layouts[i + 1];
        const z1 = l.z - totalLength / 2 + l.sizeZ / 2;
        const z2 = next.z - totalLength / 2 - next.sizeZ / 2;
        const mid = (z1 + z2) / 2;
        const len = Math.abs(z2 - z1);
        return (
          <mesh key={l.layer.id + "-link"} position={[0, 0, mid]} rotation={[Math.PI / 2, 0, 0]}>
            <cylinderGeometry args={[0.015, 0.015, len, 8]} />
            <meshBasicMaterial color="#111" transparent opacity={0.4} />
          </mesh>
        );
      })}
    </group>
  );
}

interface Architecture3DProps {
  network: Network;
  activeIndex: number;
  onActiveChange: (i: number) => void;
}

export function Architecture3D({ network, activeIndex, onActiveChange }: Architecture3DProps) {
  const [hovered, setHovered] = useState<number | null>(null);
  const layouts = useMemo(() => layoutLayers(network), [network]);
  const displayIndex = hovered ?? activeIndex;
  const displayLayer = layouts[displayIndex]?.layer;

  return (
    <div className="relative border border-foreground/10 bg-foreground/[0.02]" style={{ height: 380 }}>
      <Canvas
        camera={{ position: [3.6, 1.4, 3.4], fov: 38 }}
        dpr={[1, 2]}
        gl={{ alpha: true, antialias: true }}
      >
        <Suspense fallback={null}>
          <Scene
            network={network}
            activeIndex={activeIndex}
            setActiveIndex={onActiveChange}
            hoveredIndex={hovered}
            setHoveredIndex={setHovered}
          />
        </Suspense>
      </Canvas>

      {/* Overlay tooltip */}
      {displayLayer && (
        <div className="absolute top-3 left-3 bg-background/85 backdrop-blur-md border border-foreground/10 rounded-lg px-3 py-2 text-xs font-mono pointer-events-none max-w-[260px]">
          <div className="flex items-center justify-between gap-2 mb-1">
            <span className="font-display text-base text-foreground">{displayLayer.name}</span>
            <span
              className="w-2 h-2 rounded-full"
              style={{ background: LAYER_COLORS[displayLayer.kind] ?? "#444" }}
            />
          </div>
          <div className="text-muted-foreground">
            tensor {displayLayer.shape[0]}×{displayLayer.shape[1]}×{displayLayer.shape[2]}
          </div>
          {displayLayer.detail && (
            <div className="text-muted-foreground mt-1 leading-snug">{displayLayer.detail}</div>
          )}
          {(displayLayer.params !== undefined || displayLayer.flops !== undefined) && (
            <div className="text-muted-foreground mt-1 flex gap-3">
              {displayLayer.params !== undefined && <span>{displayLayer.params.toFixed(1)}K p</span>}
              {displayLayer.flops !== undefined && <span>{displayLayer.flops.toFixed(1)}M flop</span>}
            </div>
          )}
        </div>
      )}

      {/* Legend */}
      <div className="absolute bottom-3 left-3 right-3 flex flex-wrap items-center gap-x-4 gap-y-1 bg-background/80 backdrop-blur-md border border-foreground/10 rounded-md px-3 py-1.5 text-[10px] font-mono">
        {Object.entries(LAYER_COLORS)
          .filter(([k]) => network.layers.some((l) => l.kind === k))
          .map(([k, c]) => (
            <span key={k} className="inline-flex items-center gap-1.5 text-muted-foreground">
              <span className="w-2 h-2 rounded-sm" style={{ background: c }} />
              {k}
            </span>
          ))}
        <span className="ml-auto text-muted-foreground/70">drag mouse anywhere · click a layer to inspect</span>
      </div>
    </div>
  );
}
