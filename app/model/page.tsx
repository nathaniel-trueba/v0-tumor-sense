import type { Metadata } from "next";
import { Navigation } from "@/components/landing/navigation";
import { FooterSection } from "@/components/landing/footer-section";
import { ModelWorkshop } from "@/components/model/model-workshop";

export const metadata: Metadata = {
  title: "Model Workshop — Tumor Sense",
  description:
    "Interact with the breast tumor SVM classifier. Adjust feature sliders, swap kernels, and see the decision boundary respond in real time.",
};

export default function ModelPage() {
  return (
    <main className="relative min-h-screen overflow-x-hidden noise-overlay">
      <Navigation />
      <div className="max-w-[1400px] mx-auto px-6 lg:px-12 pt-32 pb-24 lg:pt-40 lg:pb-32">
        <ModelWorkshop />
      </div>
      <FooterSection />
    </main>
  );
}
