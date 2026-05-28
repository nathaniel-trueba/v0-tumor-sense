import type { Metadata } from "next";
import { Navigation } from "@/components/landing/navigation";
import { FooterSection } from "@/components/landing/footer-section";
import { BackgroundSection } from "@/components/research/background-section";
import { EDASection } from "@/components/research/eda-section";
import { FeatureSelectionSection } from "@/components/research/feature-selection";
import { PerspectiveSwitcher } from "@/components/research/perspective-switcher";
import { ReferencesSection } from "@/components/research/references-section";

export const metadata: Metadata = {
  title: "Research — Tumor Sense",
  description:
    "Background research, exploratory data analysis, feature selection, and ML / statistical / case-study perspectives on the breast tumor SVM classifier.",
};

export default function ResearchPage() {
  return (
    <main className="relative min-h-screen overflow-x-hidden noise-overlay">
      <Navigation />
      <div className="max-w-[1400px] mx-auto px-6 lg:px-12 pt-32 pb-24 lg:pt-40 lg:pb-32 space-y-24 lg:space-y-32">
        <BackgroundSection />
        <EDASection />
        <FeatureSelectionSection />
        <PerspectiveSwitcher />
        <ReferencesSection />
      </div>
      <FooterSection />
    </main>
  );
}
