import type { Metadata } from "next";
import { Navigation } from "@/components/landing/navigation";
import { FooterSection } from "@/components/landing/footer-section";
import { ApiDocs } from "@/components/api-docs/api-docs";

export const metadata: Metadata = {
  title: "API · Tumor Sense",
  description:
    "Reference for the Tumor Sense FastAPI server: SVM inference, image classification across five CNN/ViT architectures, and a RAG-grounded explanation endpoint.",
};

export default function ApiDocsPage() {
  return (
    <main className="relative min-h-screen overflow-x-hidden noise-overlay">
      <Navigation />
      <div className="max-w-[1400px] mx-auto px-6 lg:px-12 pt-32 pb-24 lg:pt-40 lg:pb-32">
        <ApiDocs />
      </div>
      <FooterSection />
    </main>
  );
}
