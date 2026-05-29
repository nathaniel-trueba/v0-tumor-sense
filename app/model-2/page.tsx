import type { Metadata } from "next";
import { Navigation } from "@/components/landing/navigation";
import { FooterSection } from "@/components/landing/footer-section";
import { Model2WorkshopClient } from "@/components/model-2/model2-workshop-client";

export const metadata: Metadata = {
  title: "Image model workshop · Tumor Sense",
  description:
    "Interactive breast cancer image classification workshop. Compare CNNs, ResNets, MobileNets and ViTs on real histology patches; step through the forward pass; inspect Grad-CAM, eigen-breasts and embedding-space lenses.",
};

export default function ModelTwoPage() {
  return (
    <main className="relative min-h-screen overflow-x-hidden noise-overlay">
      <Navigation />
      <div className="pt-32 pb-24">
        <div className="max-w-7xl mx-auto px-6 lg:px-12">
          <Model2WorkshopClient />
        </div>
      </div>
      <FooterSection />
    </main>
  );
}
