"use client";

import { ArrowUpRight } from "lucide-react";
import { AnimatedWave } from "./animated-wave";

const footerLinks = {
  Navigation: [
    { name: "About", href: "/#about" },
    { name: "Background", href: "/research#background" },
    { name: "Analysis", href: "/research#analysis" },
    { name: "Model", href: "/model" },
    { name: "Results", href: "/#results" },
  ],
  LinkedIn: [
    { name: "Nathaniel Trueba", href: "#developers" },
    { name: "Kavya Shah", href: "#" },
    { name: "Steven Ngo", href: "#developers" },
    { name: "Evan Park", href: "#" },
  ],
  DS3: [
    { name: "Website", href: "https://www.ds3atucsd.com" },
    { name: "Spring Projects", href: "https://www.ds3atucsd.com/projects" },
    { name: "Instagram", href: "https://www.instagram.com/ds3atucsd/" },
  ],
  HDSI: [
    { name: "Website", href: "https://datascience.ucsd.edu" },
  ],
};

const socialLinks = [
  { name: "GitHub", href: "https://github.com/VedVar43789/TumorSense" },
];

export function FooterSection() {
  return (
    <footer className="relative border-t border-foreground/10">
      {/* Animated wave background */}
      <div className="absolute inset-0 h-64 opacity-20 pointer-events-none overflow-hidden">
        <AnimatedWave />
      </div>

      <div className="relative z-10 max-w-[1400px] mx-auto px-6 lg:px-12">
        {/* Main Footer */}
        <div className="py-16 lg:py-24">
          <div className="grid grid-cols-2 md:grid-cols-6 gap-12 lg:gap-8">
            {/* Brand Column */}
            <div className="col-span-2">
              <a href="#" className="inline-flex items-center gap-2 mb-6">
                <span className="text-2xl font-display">Tumor Sense</span>
                <span className="text-xs text-muted-foreground font-mono"></span>
              </a>

              <p className="text-muted-foreground leading-relaxed mb-8 max-w-xs">
                Tumor Sense brings SVM-powered prediction to clinicians. Upload tumor characteristics and get an instant Benign or Malignant reccomendation and understand why.
              </p>

              {/* Social Links */}
              <div className="flex gap-6">
                {socialLinks.map((link) => (
                  <a
                    key={link.name}
                    href={link.href}
                    className="text-sm text-muted-foreground hover:text-foreground transition-colors flex items-center gap-1 group"
                  >
                    {link.name}
                    <ArrowUpRight className="w-3 h-3 opacity-0 -translate-x-1 group-hover:opacity-100 group-hover:translate-x-0 transition-all" />
                  </a>
                ))}
              </div>
            </div>

            {/* Link Columns */}
            {Object.entries(footerLinks).map(([title, links]) => (
              <div key={title}>
                <h3 className="text-sm font-medium mb-6">{title}</h3>
                <ul className="space-y-4">
                  {links.map((link) => (
                    <li key={link.name}>
                      <a
                        href={link.href}
                        className="text-sm text-muted-foreground hover:text-foreground transition-colors inline-flex items-center gap-2"
                      >
                        {link.name}
                      </a>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="py-8 border-t border-foreground/10 flex flex-col md:flex-row items-center justify-between gap-4">
          <p className="text-sm text-muted-foreground">
            Tumor Sense 2026. All rights reserved.
          </p>

          <div className="flex items-center gap-4 text-sm text-muted-foreground">
            <span className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-green-500" />
              All systems operational
            </span>
          </div>
        </div>
      </div>
    </footer>
  );
}
